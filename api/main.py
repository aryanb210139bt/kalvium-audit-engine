"""
api/main.py
FastAPI backend with:
  - REST endpoints for upload / status / result / download
  - URL upload  (Google Drive, Loom, direct links)
  - CSV batch upload  (multiple URLs → parallel audits)
  - WebSocket /ws/{session_id} for real-time pipeline progress
  - Static file serving for dashboard.html
  - Pipeline V3 (19-category eval, XLM sentiment, PDF/Excel reports)
"""
from __future__ import annotations
import asyncio
import csv
import io
import logging
import os
import shutil
import sqlite3
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, WebSocket, WebSocketDisconnect, Body, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config.settings import get_settings
from progress_tracker import ProgressRegistry

logger = logging.getLogger(__name__)
settings = get_settings()

app = FastAPI(title="Kalvium AI Audit Platform", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Kalvium booking authenticity routes
from api.kalvium_routes import router as kalvium_router
app.include_router(kalvium_router)

# Register deck management routes
from deck.api.routes import router as deck_router
app.include_router(deck_router)

# Register multilingual call audit v2 routes
from api.multilingual_routes import router as multilingual_router
app.include_router(multilingual_router)

# Initialise deck DB on startup
from deck.db import init_db as _init_deck_db
_init_deck_db()

# ── Startup orphan cleanup ────────────────────────────────────────────────────
import time as _time
def _sweep_orphaned_uploads():
    """Delete upload dirs older than 2 hours left by a previous crashed server."""
    try:
        cutoff = _time.time() - 2 * 3600
        base = settings.upload_dir
        if not base.exists():
            return
        for d in base.iterdir():
            if d.is_dir() and d.stat().st_mtime < cutoff:
                shutil.rmtree(d, ignore_errors=True)
                logger.info(f"Orphan cleanup: removed {d.name}")
    except Exception as e:
        logger.warning(f"Orphan sweep error: {e}")

_sweep_orphaned_uploads()

# Serve static files
_base = Path(__file__).parent.parent
static_dir = _base / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Serve logo assets
logo_dir = _base / "logo"
if logo_dir.exists():
    app.mount("/logo", StaticFiles(directory=str(logo_dir)), name="logo")


@app.get("/kalvium", include_in_schema=False)
async def kalvium_dashboard():
    return FileResponse(str(static_dir / "kalvium.html"))

_sessions: dict[str, dict] = {}
_executor = ThreadPoolExecutor(max_workers=4)
_registry = ProgressRegistry.get()

# ── Session persistence (SQLite — see session_store.py) ──────────────────────
# In-progress sessions stay in the _sessions dict above (live, ephemeral —
# that's fine to lose on a restart). The moment a session reaches a terminal
# status (completed/failed) it's written here instead, so it survives a
# restart and is safe under concurrent writes from multiple pipeline workers
# — the old data/session_history.json + data/reports/{id}.json pair could
# silently lose a write if two audits finished in the same second.
import json as _json
import session_store
session_store.init_db()
session_store.migrate_from_json()   # one-time import from the old JSON files; idempotent, safe on every startup

import activity_log
activity_log.init_db()

import auth
auth.init_db()

import associate_roster
associate_roster.init_db()

import video_audit_store
video_audit_store.init_db()

import job_queue
job_queue.init_db()
_recovered = job_queue.recover_orphaned()
if _recovered:
    logger.info(f"Recovered {_recovered} job(s) orphaned by a previous restart — reset to 'queued'")

# ── Database backend startup check (sqlite | postgres) ───────────────────────
# Never logs DATABASE_URL — only the backend name and a fixed, generic
# connectivity message. See db/backend.py, db/pg.py.
from db.backend import is_postgres_enabled, get_database_url, database_configured
if is_postgres_enabled():
    from db.pg import health_check as _db_health_check
    _pg_ok, _pg_msg = _db_health_check(get_database_url())
    (logger.info if _pg_ok else logger.error)(f"[db] backend=postgres check={_pg_msg}")
    if not _pg_ok:
        logger.error("[db] DB_BACKEND=postgres but PostgreSQL is not reachable at startup — "
                     "requests touching the database will fail until this is fixed.")
else:
    logger.info(
        f"[db] backend=sqlite (DATABASE_URL {'is' if database_configured() else 'is not'} configured, "
        f"but DB_BACKEND is not 'postgres' so it is not used)"
    )

# ── Object storage backend startup check (local | r2) — independent of the
# database backend check above. Never logs R2 credentials — only the
# backend name and a fixed, generic connectivity message. See
# storage/backend.py, storage/r2.py.
from storage.backend import is_r2_enabled, r2_configured
if is_r2_enabled():
    from storage.r2 import health_check as _r2_health_check
    _r2_ok, _r2_msg = _r2_health_check()
    (logger.info if _r2_ok else logger.error)(f"[storage] backend=r2 check={_r2_msg}")
    if not _r2_ok:
        logger.error("[storage] STORAGE_BACKEND=r2 but R2 is not reachable at startup — "
                     "persistent-file writes (video screenshots, Excel tracker, Google "
                     "Sheets auth) will still succeed locally, but each R2 sync attempt "
                     "will keep failing (logged individually) until this is fixed.")
else:
    logger.info(
        f"[storage] backend=local (R2 {'is' if r2_configured() else 'is not'} configured, "
        f"but STORAGE_BACKEND is not 'r2' so it is not used)"
    )


# ── Auth dependencies ──────────────────────────────────────────────────────────

def get_current_user(request: Request) -> dict:
    token = request.cookies.get(auth.SESSION_COOKIE)
    user = auth.get_session_user(token) if token else None
    if not user:
        raise HTTPException(401, "Not logged in")
    return user


def require_admin(user: dict = Depends(get_current_user)) -> dict:
    if user["role"] != "admin":
        raise HTTPException(403, "Admin access required")
    return user


@app.get("/api/v1/health/db")
async def health_db(admin: dict = Depends(require_admin)):
    """Admin-only. Reports which database backend is active and whether it's
    reachable — never the connection string or any credential, only a fixed
    generic status message (see db/backend.py, db/pg.py)."""
    from db.backend import is_postgres_enabled, database_configured
    backend = "postgres" if is_postgres_enabled() else "sqlite"
    if backend == "sqlite":
        return {"backend": "sqlite", "ok": True, "message": "Using local SQLite files",
                "database_url_configured": database_configured()}
    from db.backend import get_database_url
    from db.pg import health_check as _db_health_check
    ok, message = _db_health_check(get_database_url())
    return {"backend": "postgres", "ok": ok, "message": message}


@app.get("/api/v1/health/storage")
async def health_storage(admin: dict = Depends(require_admin)):
    """Admin-only. Reports which object-storage backend is active and
    whether it's reachable — never any credential, only a fixed generic
    status message (see storage/backend.py, storage/r2.py)."""
    from storage.backend import is_r2_enabled, r2_configured
    if not is_r2_enabled():
        return {"backend": "local", "ok": True, "message": "Using local disk",
                "r2_configured": r2_configured()}
    from storage.r2 import health_check as _r2_health_check
    ok, message = _r2_health_check()
    return {"backend": "r2", "ok": ok, "message": message}


def _add_to_history(session_id: str, sess: dict):
    """Persist a completed/failed session — see session_store.save_session."""
    session_store.save_session(session_id, sess)
    report = sess.get("report") or {}
    score  = report.get("score") or {}
    activity_log.log_event(
        "audit_completed", session_id=session_id, associate=sess.get("label", ""),
        actor=sess.get("actor", ""),
        detail={"overall_score": score.get("overall"), "grade": score.get("grade"),
                "source_type": sess.get("source_type", "")},
    )
    _auto_push_to_tracker(session_id, sess)


def _auto_push_to_tracker(session_id: str, sess: dict) -> None:
    """
    Audits queued from a CSV batch (lead_sheet present) push to the tracker
    automatically on completion — the whole point of a CRM import is that
    Lead Owner, TL Name, phone, etc. are already reliable, so there's
    nothing to manually review row by row. Ad-hoc single uploads (no
    lead_sheet) are unaffected — they still go through the manual "Push to
    Audit Tracker" flow, since far more fields there are guesses worth a
    human look before they land in a shared spreadsheet.
    """
    if not sess.get("lead_sheet"):
        return
    actor = sess.get("actor", "") or "Batch auto-push"
    try:
        from reports.audit_excel_manager import auto_fill_from_report, append_audit_row
        session_meta = {
            "source_url": sess.get("source_url", ""),
            "label":      sess.get("label", ""),
            "lead_sheet": sess.get("lead_sheet") or {},
            "video_analysis": video_audit_store.get_by_session(session_id),
        }
        row = auto_fill_from_report(sess.get("report") or {}, session_meta)

        changes = activity_log.diff_tracker_push(session_id, row)
        row_num = append_audit_row(row)
        activity_log.log_event(
            "pushed_to_tracker", session_id=session_id, associate=row.get("Lead Owner", ""),
            actor=actor, detail={"destination": "excel", "fields": row, "changes": changes,
                                  "row": row_num, "auto": True},
        )
    except Exception as exc:
        logger.warning(f"Auto-push to Excel tracker failed for {session_id}: {exc}")
        return   # don't attempt Sheets if even the row-building step failed

    try:
        from reports.google_sheets_manager import has_token, append_row_to_sheet
        if has_token():
            result = append_row_to_sheet(row)
            activity_log.log_event(
                "pushed_to_tracker", session_id=session_id, associate=row.get("Lead Owner", ""),
                actor=actor, detail={"destination": "gsheet", "fields": row, "changes": {},
                                      "auto": True, **result},
            )
    except Exception as exc:
        logger.warning(f"Auto-push to Google Sheet failed for {session_id}: {exc}")


def _log_audit_failed(session_id: str, sess: dict, error: str) -> None:
    activity_log.log_event(
        "audit_failed", session_id=session_id, associate=(sess or {}).get("label", ""),
        actor=(sess or {}).get("actor", ""), detail={"error": str(error)[:300]},
    )


def _load_cached_report(session_id: str) -> Optional[dict]:
    """Look up a session's full report when it's no longer in memory (e.g.
    after a restart) — see session_store.get_report."""
    return session_store.get_report(session_id)

ALLOWED_EXTENSIONS = {".mp4", ".mp3", ".wav", ".m4a", ".ogg", ".webm", ".mov"}


class SessionStatus(BaseModel):
    session_id: str
    status: str
    created_at: str
    completed_at: Optional[str] = None
    recording_file: Optional[str] = None
    error: Optional[str] = None
    video_audit_id: Optional[str] = None  # set only when analyze_video=True was requested


class UrlUploadRequest(BaseModel):
    url: str
    label: Optional[str] = None   # optional human label (e.g. counsellor name)
    actor: Optional[str] = None   # who's running this audit — activity log attribution
    analyze_video: Optional[bool] = False  # opt-in: also run Video Snapshot & Participant
                                            # Detection alongside the normal audio audit


class LinkItem(BaseModel):
    url: str
    label: Optional[str] = None


class MultiLinkRequest(BaseModel):
    links: list[LinkItem]          # 1–20 links pasted from UI
    actor: Optional[str] = None    # who's running this batch — activity log attribution
    analyze_video: Optional[bool] = False  # opt-in: also run Video Snapshot & Participant
                                            # Detection for every link in this batch


class TranscriptUploadRequest(BaseModel):
    transcript: str
    label: Optional[str] = None


class BatchStatus(BaseModel):
    batch_id: str
    total: int
    completed: int
    failed: int
    sessions: list[dict]
    created_at: str


# batch_id → list of session_ids
_batches: dict[str, dict] = {}


# ── Routes ─────────────────────────────────────────────────────────────────────

# Both routes below explicitly accept HEAD as well as GET. Render's deploy
# health-check probe sends a HEAD request — Starlette does NOT auto-add
# HEAD support to a plain @app.get(...) route (confirmed empirically: a
# GET-only route 405s on HEAD), and Render was probing "/" specifically
# with HEAD and getting exactly that 405, which is why the deploy never
# went healthy. Declaring both methods here fixes it regardless of which
# path Render's dashboard is actually configured to check.
@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    dashboard = static_dir / "dashboard.html"
    if dashboard.exists():
        return FileResponse(str(dashboard))
    return {"message": "Demo Audit API v2 — open /static/dashboard.html"}


@app.api_route("/health", methods=["GET", "HEAD"])
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat(), "sessions": len(_sessions)}


# ═══════════════════════════════════════════════════════════════════════════════
# ── AUTH ────────────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

class LoginRequest(BaseModel):
    email: str
    password: str


@app.post("/api/v1/auth/login")
async def login(req: LoginRequest, response: Response):
    user = auth.authenticate(req.email, req.password)
    if not user:
        raise HTTPException(401, "Incorrect email or password")
    token = auth.create_session(user["user_id"])
    response.set_cookie(
        auth.SESSION_COOKIE, token, httponly=True, samesite="lax",
        max_age=auth.SESSION_TTL_DAYS * 86400, path="/",
    )
    return {"status": "ok", "user": user}


@app.post("/api/v1/auth/logout")
async def logout(request: Request, response: Response):
    token = request.cookies.get(auth.SESSION_COOKIE)
    if token:
        auth.delete_session(token)
    response.delete_cookie(auth.SESSION_COOKIE, path="/")
    return {"status": "ok"}


@app.get("/api/v1/auth/me")
async def me(user: dict = Depends(get_current_user)):
    return user


# ── Admin: manage users ("granting access") ────────────────────────────────────

class CreateUserRequest(BaseModel):
    email: str
    name: str
    password: str
    role: str = "associate"
    associate_name: str = ""


class UpdateUserRequest(BaseModel):
    name: Optional[str] = None
    role: Optional[str] = None
    associate_name: Optional[str] = None
    status: Optional[str] = None
    password: Optional[str] = None


@app.get("/api/v1/admin/users")
async def admin_list_users(admin: dict = Depends(require_admin)):
    return {"users": auth.list_users()}


@app.post("/api/v1/admin/users")
async def admin_create_user(req: CreateUserRequest, admin: dict = Depends(require_admin)):
    try:
        user = auth.create_user(req.email, req.name, req.password, req.role, req.associate_name)
    except sqlite3.IntegrityError:
        raise HTTPException(409, f"An account with email {req.email!r} already exists")
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"status": "created", "user": user}


@app.patch("/api/v1/admin/users/{user_id}")
async def admin_update_user(user_id: str, req: UpdateUserRequest, admin: dict = Depends(require_admin)):
    if not auth.get_user(user_id):
        raise HTTPException(404, "User not found")
    if req.role and req.role not in ("admin", "associate"):
        raise HTTPException(400, "role must be 'admin' or 'associate'")
    if req.status and req.status not in ("active", "disabled"):
        raise HTTPException(400, "status must be 'active' or 'disabled'")
    auth.update_user(user_id, name=req.name, role=req.role,
                      associate_name=req.associate_name, status=req.status)
    if req.password:
        auth.reset_password(user_id, req.password)
    return {"status": "updated", "user": auth.get_user(user_id)}


# ── Associate roster (the master pick-list used while auditing) ────────────────

class RosterAddRequest(BaseModel):
    name: str


@app.get("/api/v1/associate-roster")
async def get_roster(user: dict = Depends(get_current_user)):
    """Any logged-in user can read the roster — they need it for the picker
    while auditing. Only admins can add/remove names (below)."""
    return {"names": associate_roster.list_names()}


@app.post("/api/v1/associate-roster")
async def add_roster_name(req: RosterAddRequest, admin: dict = Depends(require_admin)):
    if not associate_roster.add_name(req.name):
        raise HTTPException(400, "Name cannot be empty")
    return {"status": "added", "names": associate_roster.list_names()}


@app.delete("/api/v1/associate-roster/{name}")
async def remove_roster_name(name: str, admin: dict = Depends(require_admin)):
    associate_roster.remove_name(name)
    return {"status": "removed", "names": associate_roster.list_names()}


# ── Upload → Queue → manual Start ───────────────────────────────────────────
# UPLOAD creates a persisted job_queue row and does nothing else — no
# download, no FFmpeg, no Sarvam/OpenAI call. Only an explicit
# /api/v1/queue/{job_id}/start (or /start-all) hands the stored payload to
# the same run functions every upload used to call immediately. See
# job_queue.py for why: uploading must never itself consume STT/GPT-4o/API
# cost — only clicking Start does.

def _start_queued_job(job_id: str, loop) -> bool:
    """
    Reconstruct the _sessions entry for a queued job exactly as the old
    immediate-start endpoints used to build it, then submit it to the
    existing executor — the same run function as before, just triggered
    later instead of at upload time. Returns False if the job wasn't
    actually startable (already started, removed, unknown type, etc).
    """
    job = job_queue.get(job_id)
    if not job or job["status"] != "queued":
        return False

    payload = job["payload"]
    source_type = job["source_type"]
    label = job["label"]
    actor = job["actor"]
    job_queue.mark_starting(job_id)

    if source_type == "file":
        recording_path = Path(payload["recording_path"])
        _sessions[job_id] = {
            "session_id": job_id, "status": "starting",
            "created_at": job["created_at"], "recording_file": str(recording_path),
            "source_type": "file", "actor": actor,
        }
        tracker = _registry.create(job_id)
        loop.run_in_executor(_executor, _run_pipeline, job_id, recording_path, tracker)

    elif source_type == "transcript":
        _sessions[job_id] = {
            "session_id": job_id, "status": "starting",
            "created_at": job["created_at"], "label": label,
            "source_type": "transcript", "actor": actor,
        }
        tracker = _registry.create(job_id)
        loop.run_in_executor(_executor, _run_transcript_pipeline, job_id, payload["transcript"], label, tracker)

    elif source_type in ("url", "csv_row"):
        url = payload["url"]
        upload_dir = settings.upload_dir / job_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        analyze_video = bool(payload.get("analyze_video"))

        _sessions[job_id] = {
            "session_id": job_id, "status": "starting",
            "created_at": job["created_at"], "recording_file": None,
            "source_url": url, "label": label,
            "source_type": source_type, "actor": actor,
        }
        if payload.get("lead_sheet") is not None:
            _sessions[job_id]["lead_sheet"] = payload["lead_sheet"]
        if payload.get("batch_id"):
            _sessions[job_id]["batch_id"] = payload["batch_id"]

        tracker = _registry.create(job_id)

        if analyze_video:
            video_audit_id = str(uuid.uuid4())
            video_audit_store.create(
                video_audit_id, linked_session_id=job_id, source_url=url,
                label=label, actor=actor, created_at=datetime.utcnow().isoformat(),
            )
            activity_log.log_event("video_analysis_queued", session_id=job_id, associate=label,
                                    actor=actor, detail={"video_audit_id": video_audit_id})
            _sessions[job_id]["video_audit_id"] = video_audit_id
            loop.run_in_executor(_executor, _run_preprocessed_pipeline, job_id, url, upload_dir, tracker,
                                  video_audit_id, label, actor)
        elif source_type == "csv_row":
            # CSV rows historically use the full-download path, not the
            # audio-only streaming path — matches pre-existing behavior.
            loop.run_in_executor(_executor, _run_url_pipeline, job_id, url, upload_dir, tracker)
        else:
            loop.run_in_executor(_executor, _run_url_pipeline_audio, job_id, url, upload_dir, tracker)

    else:
        job_queue.mark_failed(job_id, f"Unknown source_type: {source_type}", datetime.utcnow().isoformat())
        return False

    activity_log.log_event("audit_started", session_id=job_id, associate=label, actor=actor,
                            detail={"source_type": source_type})
    return True


@app.get("/api/v1/queue")
async def get_queue():
    """Everything not yet in a terminal state, for the Upload/Queue and
    Upfront Auditing sections. 'processing' items also have a live WS
    connection available at /ws/{job_id}, exactly as before."""
    active = job_queue.list_active()
    return {
        "queued": [j for j in active if j["status"] in ("queued", "starting")],
        "processing": [j for j in active if j["status"] == "processing"],
    }


@app.post("/api/v1/queue/{job_id}/start")
async def start_queued_job(job_id: str):
    loop = asyncio.get_running_loop()
    ok = _start_queued_job(job_id, loop)
    if not ok:
        raise HTTPException(400, "Job not found or not in a startable state")
    return {"status": "starting", "job_id": job_id}


@app.post("/api/v1/queue/start-all")
async def start_all_queued():
    """
    Submits every currently-queued job to the existing ThreadPoolExecutor at
    once. This is safe without any new capacity-tracking code: each run
    function only flips its job_queue status to 'processing' once a worker
    thread actually begins executing it (see job_queue.mark_processing calls
    throughout the run functions) — a job still waiting behind the
    executor's own concurrency limit correctly stays 'starting' until then.
    """
    loop = asyncio.get_running_loop()
    jobs = job_queue.list_queued_only()
    started = [j["job_id"] for j in jobs if _start_queued_job(j["job_id"], loop)]
    return {"started": started, "count": len(started)}


@app.delete("/api/v1/queue/{job_id}")
async def remove_queued_job(job_id: str):
    """Only removes a job still in 'queued' — matches the spec's rule that
    processing/completed jobs are never silently deleted this way."""
    ok = job_queue.remove(job_id)
    if not ok:
        raise HTTPException(400, "Job not queued (already started, or doesn't exist)")
    return {"status": "removed", "job_id": job_id}


@app.post("/api/v1/queue/{job_id}/retry")
async def retry_queued_job(job_id: str):
    """Re-queues a failed job using its originally stored payload — no
    re-upload needed. Full re-run only (no partial-resume of a previously
    successful preprocessing stage exists yet)."""
    ok = job_queue.retry(job_id, datetime.utcnow().isoformat())
    if not ok:
        raise HTTPException(400, "Job not in a failed state")
    return {"status": "queued", "job_id": job_id}


@app.post("/api/v1/queue/{job_id}/cancel")
async def cancel_processing_job(job_id: str, actor: str = ""):
    """
    Soft-cancel a 'starting'/'processing' job — hides it from Upfront
    Auditing immediately. Does NOT stop the background pipeline thread (no
    safe way to interrupt mid-transcription without touching the protected
    audit/STT pipeline), so it may still complete and land in Audit
    History/the tracker later — this only removes it from view here.
    """
    ok = job_queue.cancel(job_id, datetime.utcnow().isoformat())
    if not ok:
        raise HTTPException(400, "Job not in a cancellable state")
    job = job_queue.get(job_id)
    if job:
        activity_log.log_event("audit_cancelled", session_id=job_id, associate=job.get("label", ""),
                                actor=actor, detail={"note": "Soft-cancelled — background job may still complete"})
    return {"status": "cancelled", "job_id": job_id}


@app.post("/api/v1/audit/upload", response_model=SessionStatus)
async def upload_recording(file: UploadFile = File(...), actor: str = Form("")):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported type '{ext}'")

    content = await file.read()
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(413, f"File too large")

    session_id = str(uuid.uuid4())
    upload_dir = settings.upload_dir / session_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    recording_path = upload_dir / f"recording{ext}"
    recording_path.write_bytes(content)

    created_at = datetime.utcnow().isoformat()
    job_queue.enqueue(session_id, "file", label=Path(file.filename or "recording").stem, actor=actor,
                       payload={"recording_path": str(recording_path)}, created_at=created_at,
                       filename=file.filename or "")
    activity_log.log_event("uploaded", session_id=session_id, actor=actor,
                            detail={"source_type": "file", "filename": file.filename,
                                    "size_kb": len(content) // 1024})

    logger.info(f"Session {session_id} queued (not started) — {file.filename} ({len(content)//1024}KB)")
    return SessionStatus(session_id=session_id, status="queued", created_at=created_at)


@app.post("/api/v1/audit/upload-url", response_model=SessionStatus)
async def upload_from_url(req: UrlUploadRequest):
    """
    Download a recording from a URL and run the audit pipeline.
    Supports: Google Drive, Loom, any direct .mp4/.mp3/.wav/.m4a/.webm link.
    """
    from utils.url_downloader import validate_url, preflight_drive, detect_source

    url = req.url.strip()
    if not url:
        raise HTTPException(400, "URL is required")

    try:
        validate_url(url)         # cheap HEAD check — no download, no AI cost
        preflight_drive(url)      # catches restricted Drive files before queuing
    except ValueError as e:
        raise HTTPException(400, str(e))

    session_id = str(uuid.uuid4())
    label = req.label or url[:60]
    created_at = datetime.utcnow().isoformat()

    job_queue.enqueue(session_id, "url", label=label, actor=req.actor or "",
                       payload={"url": url, "analyze_video": bool(req.analyze_video)},
                       created_at=created_at, filename=url[:200])
    activity_log.log_event("uploaded", session_id=session_id, associate=label, actor=req.actor or "",
                            detail={"source_type": "link", "url": url[:200]})

    logger.info(f"Job {session_id} queued from URL (not started) — {url[:60]}")
    return SessionStatus(session_id=session_id, status="queued", created_at=created_at)


@app.post("/api/v1/audit/upload-links")
async def upload_multiple_links(req: MultiLinkRequest):
    """
    Submit 1–20 Google Drive / Loom / direct links in a single call.
    Each link becomes its own session processed sequentially in the background.
    Returns a batch_id plus the list of queued session_ids.
    """
    from utils.url_downloader import validate_url, detect_source, preflight_drive

    if not req.links:
        raise HTTPException(400, "No links provided")
    if len(req.links) > 20:
        raise HTTPException(400, "Max 20 links per batch")

    batch_id    = str(uuid.uuid4())
    session_ids = []
    errors      = []

    for i, item in enumerate(req.links):
        url = item.url.strip()
        if not url:
            errors.append({"index": i, "error": "Empty URL — skipped"})
            continue

        try:
            validate_url(url)         # cheap HEAD check only — no download, no AI cost
            preflight_drive(url)
        except ValueError as e:
            errors.append({"index": i, "url": url[:80], "error": str(e)})
            continue

        session_id = str(uuid.uuid4())
        label = item.label or detect_source(url).upper() + f" #{i+1}"
        created_at = datetime.utcnow().isoformat()

        job_queue.enqueue(session_id, "url", label=label, actor=req.actor or "",
                           payload={"url": url, "analyze_video": bool(req.analyze_video), "batch_id": batch_id},
                           created_at=created_at, filename=url[:200], batch_id=batch_id)
        activity_log.log_event("uploaded", session_id=session_id, associate=label, actor=req.actor or "",
                                detail={"source_type": "link", "url": url[:200], "batch_id": batch_id})
        session_ids.append(session_id)

    _batches[batch_id] = {
        "batch_id":    batch_id,
        "total":       len(session_ids),
        "session_ids": session_ids,
        "errors":      errors,
        "created_at":  datetime.utcnow().isoformat(),
    }

    logger.info(f"Links batch {batch_id}: {len(session_ids)} queued (not started), {len(errors)} skipped")
    return {
        "batch_id":    batch_id,
        "queued":      len(session_ids),
        "skipped":     len(errors),
        "session_ids": session_ids,
        "errors":      errors,
    }


@app.post("/api/v1/audit/upload-transcript", response_model=SessionStatus)
async def upload_transcript(file: UploadFile = File(...), actor: str = Form("")):
    """
    Accept a .txt transcript file and run the full audit pipeline,
    skipping STT entirely — goes straight to GPT evaluation + deck coverage.

    Transcript format (each line):
        Counsellor: text
        Student: text
    """
    content = await file.read()
    try:
        raw_text = content.decode("utf-8")
    except UnicodeDecodeError:
        raw_text = content.decode("latin-1")

    if not raw_text.strip():
        raise HTTPException(400, "Transcript file is empty")

    label = Path(file.filename or "transcript").stem
    session_id = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat()

    job_queue.enqueue(session_id, "transcript", label=label, actor=actor,
                       payload={"transcript": raw_text}, created_at=created_at,
                       filename=file.filename or "")
    activity_log.log_event("uploaded", session_id=session_id, associate=label, actor=actor,
                            detail={"source_type": "transcript", "filename": file.filename})

    logger.info(f"Job {session_id} queued from transcript file '{file.filename}' (not started, {len(raw_text)} chars)")
    return SessionStatus(session_id=session_id, status="queued", created_at=created_at)


VIDEO_ANALYSIS_BATCH_CAP = 20  # safety cap: analyze_video does a full video
                                # download + 5 GPT-4o vision calls per row —
                                # real cost — so it's capped much lower than
                                # the 1000-row audio-only batch limit.


@app.post("/api/v1/audit/upload-csv")
async def upload_csv_batch(file: UploadFile = File(...), actor: str = Form(""),
                            analyze_video: bool = Form(False)):
    """
    Upload a CSV file containing meeting recording URLs — this is also the
    "lead sheet" join. The URL and associate-name columns are detected from
    a flexible set of header names (below); every OTHER column in the file
    rides along unchanged as CRM data — it auto-fills any Audit Tracker
    column with a matching name (case-insensitively; see auto_fill_from_report
    in reports/audit_excel_manager.py) and is otherwise preserved as-is and
    shown in full on that audit's detail page. Nothing in the file is dropped.

    URL column     (required):  url | link | recording_url | recording | demo link
    Associate name (optional):  label | associate | name | counsellor | title | lead owner

    Returns a batch_id and the list of session_ids queued.
    """
    from utils.url_downloader import validate_url, detect_source

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "File must be a .csv")

    content = await file.read()
    if len(content) > 5 * 1024 * 1024:   # 5MB CSV limit
        raise HTTPException(413, "CSV too large (max 5MB)")

    text = content.decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(text))

    # Find the URL + associate-name columns (flexible header names); every
    # other column passes through untouched as CRM lead-sheet data.
    url_col = label_col = None
    rows = []
    for row in reader:
        if url_col is None:
            lower = {k.lower().strip(): k for k in row.keys()}
            url_col   = (lower.get("demo link") or lower.get("url") or lower.get("link")
                         or lower.get("recording_url") or lower.get("recording"))
            label_col = (lower.get("lead owner") or lower.get("label") or lower.get("associate")
                         or lower.get("counsellor") or lower.get("name") or lower.get("title"))
            if not url_col:
                raise HTTPException(400,
                    "CSV must have a column named 'url', 'link', 'demo link', or 'recording_url'")
        rows.append(row)

    if not rows:
        raise HTTPException(400, "CSV has no data rows")
    if len(rows) > 1000:
        raise HTTPException(400, "Max 1000 recordings per batch")
    if analyze_video and len(rows) > VIDEO_ANALYSIS_BATCH_CAP:
        raise HTTPException(400,
            f"Video analysis is opt-in per batch and capped at {VIDEO_ANALYSIS_BATCH_CAP} rows "
            f"(you have {len(rows)}) — it downloads the full video and runs 5 GPT-4o vision "
            f"calls per row, real cost per row. Split into smaller batches, or upload without "
            f"video analysis and run it per-audit afterward from the audit's detail page.")

    roster_lower = {n.lower() for n in associate_roster.list_names()}

    batch_id = str(uuid.uuid4())
    session_ids = []
    errors = []
    off_roster: set[str] = set()

    for i, row in enumerate(rows):
        url = row.get(url_col, "").strip()
        label = (row.get(label_col, "") if label_col else "").strip() or f"Row {i+1}"
        if label_col and row.get(label_col, "").strip() and label.lower() not in roster_lower:
            off_roster.add(label)

        if not url:
            errors.append({"row": i + 1, "error": "Empty URL — skipped"})
            continue

        try:
            validate_url(url)   # cheap HEAD check only — no download, no AI cost
        except ValueError as e:
            errors.append({"row": i + 1, "url": url[:60], "error": str(e)})
            continue

        session_id = str(uuid.uuid4())
        created_at = datetime.utcnow().isoformat()

        # Every column except the URL itself — the full CRM row, untouched.
        lead_sheet = {k: v for k, v in row.items() if k != url_col}

        job_queue.enqueue(session_id, "csv_row", label=label, actor=actor,
                           payload={"url": url, "analyze_video": analyze_video,
                                    "lead_sheet": lead_sheet, "batch_id": batch_id},
                           created_at=created_at, filename=url[:200], batch_id=batch_id)
        activity_log.log_event("uploaded", session_id=session_id, associate=label, actor=actor,
                                detail={"source_type": "csv", "url": url[:200], "batch_id": batch_id, "row": i + 1})
        session_ids.append(session_id)

    _batches[batch_id] = {
        "batch_id": batch_id,
        "total": len(session_ids),
        "session_ids": session_ids,
        "errors": errors,
        "created_at": datetime.utcnow().isoformat(),
    }

    logger.info(f"Batch {batch_id}: {len(session_ids)} rows queued (not started), {len(errors)} skipped")
    return {
        "batch_id": batch_id,
        "queued": len(session_ids),
        "skipped": len(errors),
        "session_ids": session_ids,
        "errors": errors,
        "off_roster_names": sorted(off_roster),   # associate names in the file not on the roster — not blocked, just flagged
    }


@app.get("/api/v1/batch/{batch_id}")
async def get_batch_status(batch_id: str):
    """Poll status of all sessions in a batch."""
    b = _batches.get(batch_id)
    if not b:
        raise HTTPException(404, "Batch not found")

    sessions_info = []
    completed = failed = 0
    for sid in b["session_ids"]:
        s = _sessions.get(sid, {})
        status = s.get("status")
        if status is None:
            # Not started yet — this session only exists in job_queue so far.
            job = job_queue.get(sid)
            status = job["status"] if job else "unknown"
        if status == "completed":
            completed += 1
        elif status == "failed":
            failed += 1
        sessions_info.append({
            "session_id": sid,
            "label": s.get("label", sid[:8]),
            "status": status,
            "source_url": s.get("source_url", ""),
            "score": s["report"]["score"]["overall"] if status == "completed" and s.get("report") else None,
            "grade": s["report"]["score"]["grade"] if status == "completed" and s.get("report") else None,
            "error": s.get("error"),
        })

    return {
        "batch_id": batch_id,
        "total": b["total"],
        "completed": completed,
        "failed": failed,
        "in_progress": b["total"] - completed - failed,
        "sessions": sessions_info,
        "created_at": b["created_at"],
    }


@app.get("/api/v1/audit/{session_id}/status", response_model=SessionStatus)
async def get_status(session_id: str):
    s = _sessions.get(session_id)
    if not s:
        raise HTTPException(404, "Session not found")
    return SessionStatus(**{k: v for k, v in s.items() if k != "report"})


@app.get("/api/v1/audit/{session_id}")
async def get_result(session_id: str):
    s = _sessions.get(session_id)
    if s:
        if s["status"] != "completed":
            return {"session_id": session_id, "status": s["status"]}
        return s["report"]
    # Not in memory (e.g. server restarted since this session completed) —
    # fall back to session_store (SQLite).
    cached = _load_cached_report(session_id)
    if cached:
        return cached
    raise HTTPException(404, "Session not found — may have been cleared from memory")


@app.get("/api/v1/audit/{session_id}/participation")
async def get_participation(session_id: str):
    s = _sessions.get(session_id)
    if not s:
        raise HTTPException(404, "Session not found")
    if s["status"] != "completed":
        raise HTTPException(409, "Audit not yet complete")
    pi = (s.get("report") or {}).get("participant_intelligence")
    if not pi:
        raise HTTPException(404, "Participant intelligence not available for this session")
    return pi


@app.get("/api/v1/audit/{session_id}/attendance")
async def get_attendance(session_id: str):
    s = _sessions.get(session_id)
    if not s:
        raise HTTPException(404, "Session not found")
    if s["status"] != "completed":
        raise HTTPException(409, "Audit not yet complete")
    pi = (s.get("report") or {}).get("participant_intelligence")
    if not pi:
        raise HTTPException(404, "Participant intelligence not available for this session")
    return pi.get("attendance", {})


@app.get("/api/v1/audit/{session_id}/participants")
async def get_participants(session_id: str):
    s = _sessions.get(session_id)
    if not s:
        raise HTTPException(404, "Session not found")
    if s["status"] != "completed":
        raise HTTPException(409, "Audit not yet complete")
    pi = (s.get("report") or {}).get("participant_intelligence")
    if not pi:
        raise HTTPException(404, "Participant intelligence not available for this session")
    return pi.get("participants", [])


@app.get("/api/v1/sessions")
async def list_sessions(limit: int = 20):
    sessions = sorted(_sessions.values(), key=lambda s: s.get("created_at", ""), reverse=True)[:limit]
    return [
        {
            "session_id": s["session_id"],
            "status": s["status"],
            "created_at": s.get("created_at"),
            "score": s["report"]["score"]["overall"] if s.get("report") else None,
            "grade": s["report"]["score"]["grade"] if s.get("report") else None,
        }
        for s in sessions
    ]


# ── Download endpoints ──────────────────────────────────────────────────────────

def _get_report_for_download(session_id: str) -> dict:
    """Full report for a download endpoint — checks live memory first, then
    falls back to session_store (so downloads still work after a restart,
    same fix as get_result/reload)."""
    s = _sessions.get(session_id)
    if s:
        if s["status"] != "completed":
            raise HTTPException(409, f"Report not ready — status: {s['status']}")
        return s["report"], s.get("label", "")
    cached = session_store.get_session(session_id)
    if cached and cached.get("report"):
        return cached["report"], cached.get("label", "")
    raise HTTPException(404, "Session not found")


@app.get("/api/v1/audit/{session_id}/download/pdf")
async def download_pdf(session_id: str, actor: str = ""):
    """Download full audit report as enterprise-grade PDF."""
    report, label = _get_report_for_download(session_id)
    try:
        from reports.pdf_generator import generate_pdf
        pdf_bytes = await asyncio.get_running_loop().run_in_executor(
            _executor, generate_pdf, report
        )
        filename = f"kalvium_audit_{session_id[:8]}_{datetime.utcnow().strftime('%Y%m%d')}.pdf"
        activity_log.log_event("pdf_downloaded", session_id=session_id, associate=label, actor=actor)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except Exception as exc:
        logger.exception(f"PDF generation failed for {session_id}: {exc}")
        raise HTTPException(500, f"PDF generation failed: {str(exc)[:200]}")


@app.get("/api/v1/audit/{session_id}/download/excel")
async def download_excel(session_id: str, actor: str = ""):
    """Download full audit report as multi-sheet Excel workbook."""
    report, label = _get_report_for_download(session_id)
    try:
        from reports.excel_generator import generate_excel
        excel_bytes = await asyncio.get_running_loop().run_in_executor(
            _executor, generate_excel, report
        )
        filename = f"kalvium_audit_{session_id[:8]}_{datetime.utcnow().strftime('%Y%m%d')}.xlsx"
        activity_log.log_event("excel_downloaded", session_id=session_id, associate=label, actor=actor)
        return Response(
            content=excel_bytes,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except Exception as exc:
        logger.exception(f"Excel generation failed for {session_id}: {exc}")
        raise HTTPException(500, f"Excel generation failed: {str(exc)[:200]}")


@app.get("/api/v1/audit/{session_id}/download/transcript")
async def download_transcript(session_id: str, actor: str = ""):
    """Download plain-text annotated transcript."""
    report, label = _get_report_for_download(session_id)
    activity_log.log_event("transcript_downloaded", session_id=session_id, associate=label, actor=actor)
    lines = [
        f"KALVIUM DEMO AUDIT — TRANSCRIPT",
        f"Session: {session_id[:8].upper()}",
        f"Date: {report.get('created_at', '')[:19]}",
        f"Language: {report.get('language_detected', '').upper()}",
        f"Duration: {int(report.get('duration_seconds', 0)//60)}m {int(report.get('duration_seconds', 0)%60)}s",
        f"Score: {report['score']['overall']}/100 ({report['score']['grade']})",
        "=" * 80, "",
    ]
    for utt in report.get("utterances", []):
        start = utt.get("start_time", 0) if isinstance(utt, dict) else utt.start_time
        m, sc = divmod(int(start), 60)
        sp = utt.get("speaker", "?") if isinstance(utt, dict) else utt.speaker.value
        native  = utt.get("native_text", "") if isinstance(utt, dict) else utt.native_text
        english = utt.get("english_text", "") if isinstance(utt, dict) else utt.english_text
        lang    = utt.get("language_detected", "en") if isinstance(utt, dict) else utt.language_detected
        lines.append(f"[{m:02d}:{sc:02d}] {sp} ({lang})")
        lines.append(f"  {native}")
        if english and english != native:
            lines.append(f"  → {english}")
        lines.append("")

    text = "\n".join(lines)
    filename = f"transcript_{session_id[:8]}.txt"
    return Response(
        content=text.encode("utf-8"),
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── WebSocket — live pipeline stream ───────────────────────────────────────────

@app.websocket("/ws/{session_id}")
async def ws_progress(ws: WebSocket, session_id: str):
    await ws.accept()
    tracker = _registry.lookup(session_id)
    if not tracker:
        await ws.send_json({"type": "error", "message": "Session not found"})
        await ws.close()
        return

    # Replay history for reconnects
    for event in list(tracker.history):
        try:
            await ws.send_json(event)
        except Exception:
            return

    loop = asyncio.get_running_loop()
    try:
        while True:
            event = await loop.run_in_executor(None, tracker.get_event, 1.0)
            if event is None:
                break
            if event == "TIMEOUT":
                try:
                    await ws.send_json({"type": "ping"})
                except Exception:
                    break
                continue
            try:
                await ws.send_json(event)
            except (WebSocketDisconnect, Exception):
                break
    finally:
        try:
            await ws.close()
        except Exception:
            pass


# ── Pipeline runner ─────────────────────────────────────────────────────────────

def _run_pipeline(session_id: str, recording_path: Path, tracker) -> None:
    from pipeline_v3 import DemoAuditPipelineV3
    # Marks the moment a worker thread actually begins running this job —
    # not when Start was clicked. This is what makes "Start All" respect
    # existing concurrency: a job still waiting behind others in the
    # ThreadPoolExecutor's own queue hasn't reached this line yet, so it
    # correctly stays 'starting' rather than falsely showing 'processing'.
    # _run_pipeline is the convergence point for file/url/preprocessed
    # uploads, so completed/failed here covers all of them at once too.
    job_queue.mark_processing(session_id, datetime.utcnow().isoformat())
    try:
        _sessions[session_id]["status"] = "processing"
        pipeline = DemoAuditPipelineV3(progress=tracker)
        report_dict = pipeline.run(recording_path)
        _sessions[session_id].update({
            "status": "completed",
            "completed_at": datetime.utcnow().isoformat(),
            "report": report_dict,
        })
        _add_to_history(session_id, _sessions[session_id])
        job_queue.mark_completed(session_id, datetime.utcnow().isoformat())
        logger.info(f"Session {session_id} complete — {report_dict['score']['overall']}/100")
    except Exception as exc:
        logger.exception(f"Pipeline failed for {session_id}: {exc}")
        _sessions[session_id].update({"status": "failed", "error": str(exc)})
        _log_audit_failed(session_id, _sessions[session_id], str(exc))
        job_queue.mark_failed(session_id, str(exc), datetime.utcnow().isoformat())
        tracker.error(str(exc))


def _parse_transcript(raw: str) -> list[dict]:
    """
    Parse Kalvium audit transcript export format into utterance dicts.

    File format (system export):
        [MM:SS] Speaker.COUNSELLOR (te-IN)
          <native text line 1>
        <native text line 2>           ← continuation, no indent required
          → <english line 1>          ← first english line has → prefix
        <english line 2>              ← continuation english, no prefix

    Each utterance block ends when the next [timestamp] header appears.
    We use ONLY the english_text (→ lines) for evaluation.
    """
    import re

    HEADER_RE = re.compile(
        r'^\[(\d+:\d+)\]\s+Speaker\.(COUNSELLOR|STUDENT|UNKNOWN)\s*\(([^)]+)\)',
        re.IGNORECASE
    )
    SKIP_LINES = re.compile(
        r'^(KALVIUM DEMO|Session:|Date:|Language:|Duration:|Score:|=+\s*$)',
        re.IGNORECASE
    )

    # Detect system format: look for [MM:SS] Speaker. pattern
    if not re.search(r'\[\d+:\d+\]\s+Speaker\.', raw):
        # Fallback: simple "Counsellor: text" format
        return _parse_simple_transcript(raw)

    utterances      = []
    cur_speaker     = "counsellor"
    cur_native      = []
    cur_english     = []
    cur_start       = 0.0
    cur_lang        = "en"
    in_english      = False   # True once we've seen the → line
    utt_idx         = 0

    def flush():
        nonlocal utt_idx
        eng = " ".join(cur_english).strip()
        nat = " ".join(cur_native).strip()
        if not (eng or nat):
            return
        word_count = len((eng or nat).split())
        duration   = max(2.0, word_count * 0.4)
        utterances.append({
            "utterance_id":      f"utt_{utt_idx:04d}",
            "speaker":           cur_speaker,
            "english_text":      eng or nat,
            "native_text":       nat or eng,
            "start_time":        round(cur_start, 1),
            "end_time":          round(cur_start + duration, 1),
            "language_detected": cur_lang,
            "confidence":        1.0,
        })
        utt_idx += 1
        cur_native.clear()
        cur_english.clear()

    for line in raw.splitlines():
        # New utterance header
        m = HEADER_RE.match(line)
        if m:
            flush()
            in_english = False
            parts = m.group(1).split(":")
            cur_start   = int(parts[0]) * 60 + int(parts[1])
            cur_speaker = "counsellor" if "COUNSELLOR" in m.group(2).upper() else "student"
            cur_lang    = m.group(3).strip().lower()
            continue

        stripped = line.strip()
        if not stripped or SKIP_LINES.match(stripped):
            continue

        # English translation line (starts with →)
        if stripped.startswith("→"):
            in_english = True
            cur_english.append(stripped.lstrip("→").strip())
        elif in_english:
            # Continuation of english block
            cur_english.append(stripped)
        else:
            # Native text
            cur_native.append(stripped)

    flush()
    return utterances


def _parse_simple_transcript(raw: str) -> list[dict]:
    """Fallback parser for 'Counsellor: text / Student: text' format."""
    import re
    COUNSELLOR_RE = re.compile(r'^(counsello?r|agent|advisor|sales|host)\s*:', re.IGNORECASE)
    STUDENT_RE    = re.compile(r'^(student|parent|customer|prospect|client|lead)\s*:', re.IGNORECASE)

    utterances  = []
    cur_speaker = "counsellor"
    cur_lines   = []
    fake_time   = 0.0
    utt_idx     = 0

    def flush():
        nonlocal fake_time, utt_idx
        txt = " ".join(cur_lines).strip()
        if not txt: return
        duration = max(2.0, len(txt.split()) * 0.4)
        utterances.append({
            "utterance_id":    f"utt_{utt_idx:04d}",
            "speaker":         cur_speaker,
            "english_text":    txt, "native_text": txt,
            "start_time":      round(fake_time, 1),
            "end_time":        round(fake_time + duration, 1),
            "language_detected": "en", "confidence": 1.0,
        })
        fake_time += duration + 0.5
        utt_idx   += 1
        cur_lines.clear()

    for line in raw.splitlines():
        line = line.strip()
        if not line: continue
        if COUNSELLOR_RE.match(line):
            flush(); cur_speaker = "counsellor"
            cur_lines = [re.sub(r'^[^:]+:\s*', '', line)]
        elif STUDENT_RE.match(line):
            flush(); cur_speaker = "student"
            cur_lines = [re.sub(r'^[^:]+:\s*', '', line)]
        else:
            cur_lines.append(line)
    flush()
    return utterances


def _run_transcript_pipeline(session_id: str, raw_transcript: str,
                              label: str, tracker) -> None:
    """Run full audit pipeline from a pasted transcript, skipping STT."""
    from pipeline_v3 import DemoAuditPipelineV3
    job_queue.mark_processing(session_id, datetime.utcnow().isoformat())
    try:
        _sessions[session_id]["status"] = "processing"
        utterances = _parse_transcript(raw_transcript)
        if not utterances:
            raise ValueError("Could not parse any utterances from transcript. "
                             "Use format 'Counsellor: text' / 'Student: text'")

        tracker.log(f"Parsed {len(utterances)} utterances from transcript")
        pipeline = DemoAuditPipelineV3(progress=tracker)
        report_dict = pipeline.run_from_transcript(utterances, label=label)
        _sessions[session_id].update({
            "status":       "completed",
            "completed_at": datetime.utcnow().isoformat(),
            "report":       report_dict,
        })
        _add_to_history(session_id, _sessions[session_id])
        job_queue.mark_completed(session_id, datetime.utcnow().isoformat())
        logger.info(f"Session {session_id} (transcript) complete — "
                    f"{report_dict['score']['overall']}/100")
    except Exception as exc:
        logger.exception(f"Transcript pipeline failed for {session_id}: {exc}")
        _sessions[session_id].update({"status": "failed", "error": str(exc)})
        _log_audit_failed(session_id, _sessions[session_id], str(exc))
        job_queue.mark_failed(session_id, str(exc), datetime.utcnow().isoformat())
        tracker.error(str(exc))


def _run_url_pipeline(session_id: str, url: str, upload_dir: Path, tracker) -> None:
    """Download from URL then run the audit pipeline, then clean up the file."""
    from utils.url_downloader import download_recording
    job_queue.mark_processing(session_id, datetime.utcnow().isoformat())
    recording_path = None
    try:
        tracker.log("Downloading recording…")
        recording_path = download_recording(url, upload_dir)
        _sessions[session_id]["recording_file"] = str(recording_path)
        tracker.log(f"Download complete: {recording_path.name} "
                    f"({recording_path.stat().st_size // 1024}KB)")
        _run_pipeline(session_id, recording_path, tracker)
    except Exception as exc:
        logger.exception(f"URL download failed for {session_id}: {exc}")
        _sessions[session_id].update({"status": "failed", "error": str(exc)})
        _log_audit_failed(session_id, _sessions[session_id], str(exc))
        job_queue.mark_failed(session_id, str(exc), datetime.utcnow().isoformat())
        tracker.error(f"Download failed: {str(exc)[:200]}")
    finally:
        _cleanup_upload_dir(upload_dir)


def _run_url_pipeline_audio(session_id: str, url: str, upload_dir: Path, tracker) -> None:
    """
    Extract audio from URL via FFmpeg (direct URL mode) → run audit pipeline.
    FFmpeg downloads and extracts audio in one pass using its built-in HTTP client.
    Only the resulting MP3 is written to disk; cleaned up after the pipeline.
    """
    from utils.url_downloader import download_as_audio, detect_source
    job_queue.mark_processing(session_id, datetime.utcnow().isoformat())
    recording_path = None
    try:
        source = detect_source(url)
        tracker.log(f"Extracting audio from {source.upper()} via FFmpeg…")

        def _progress(msg: str):
            tracker.log(msg)

        recording_path = download_as_audio(url, upload_dir, progress_cb=_progress)
        _sessions[session_id]["recording_file"] = str(recording_path)
        size_mb = recording_path.stat().st_size / 1024 / 1024
        tracker.log(f"Audio ready: {size_mb:.1f}MB MP3 — starting pipeline…")
        _run_pipeline(session_id, recording_path, tracker)
    except Exception as exc:
        logger.exception(f"Audio pipeline failed for {session_id}: {exc}")
        _sessions[session_id].update({"status": "failed", "error": str(exc)})
        _log_audit_failed(session_id, _sessions[session_id], str(exc))
        job_queue.mark_failed(session_id, str(exc), datetime.utcnow().isoformat())
        tracker.error(f"Failed: {str(exc)[:300]}")
    finally:
        _cleanup_upload_dir(upload_dir)


def _cleanup_upload_dir(upload_dir: Path) -> None:
    """Delete the session upload directory after pipeline completes."""
    try:
        if upload_dir.exists():
            shutil.rmtree(upload_dir, ignore_errors=True)
            logger.debug(f"Cleaned up: {upload_dir}")
    except Exception as e:
        logger.warning(f"Cleanup failed for {upload_dir}: {e}")


def _persist_screenshots_to_r2(video_audit_id: str, out_dir: Path, screenshots: list[dict]) -> None:
    """
    If STORAGE_BACKEND=r2, uploads each successfully-extracted screenshot
    to R2 (key: audits/{video_audit_id}/screenshots/{filename}) and adds
    r2_key/content_type/size_bytes to that screenshot's dict IN PLACE, so
    they land in screenshots_json when video_audit_store.complete() saves
    it right after this call. Only deletes the local out_dir once every
    screenshot has been confirmed uploaded — if any upload fails, the
    local copies are left in place as a fallback and the screenshot
    endpoint below will keep serving them locally. No-op entirely when
    STORAGE_BACKEND=local (the default) — behavior is then identical to
    before this feature existed.
    """
    from storage.backend import is_r2_enabled
    if not is_r2_enabled():
        return

    from storage.r2 import upload_file, StorageError

    all_ok = True
    for shot in screenshots:
        if shot.get("extract_failed") or not shot.get("filename"):
            continue
        local_path = out_dir / shot["filename"]
        if not local_path.exists():
            continue
        key = f"audits/{video_audit_id}/screenshots/{shot['filename']}"
        try:
            upload_file(local_path, key, content_type="image/jpeg")
            shot["r2_key"] = key
            shot["content_type"] = "image/jpeg"
            shot["size_bytes"] = local_path.stat().st_size
        except StorageError as exc:
            all_ok = False
            logger.error(f"Failed to persist screenshot {shot['filename']} for "
                         f"video_audit {video_audit_id} to R2: {type(exc).__name__}: {exc}")

    if all_ok:
        try:
            shutil.rmtree(out_dir, ignore_errors=True)
        except Exception as exc:
            logger.warning(f"Could not clean up local screenshots for {video_audit_id}: {exc}")


def _preprocess_and_analyze_video(video_path: Path, video_audit_id: str, linked_session_id: str,
                                   label: str, actor: str) -> None:
    """
    Shared step, given an ALREADY-DOWNLOADED local video file: local
    metadata + screenshot extraction (FFmpeg/FFprobe/Pillow only — no AI
    API, no GPU), saved as a preprocessing manifest, then the existing,
    already-in-production GPT-4o vision pass on those screenshots (unchanged
    from before — this function just stops it from needing its own second
    download). Used by both _run_video_snapshot_job (standalone, post-hoc
    trigger — downloads its own copy first) and _run_preprocessed_pipeline
    (fresh upload — shares the one download with the audio pipeline).

    Any failure here only marks this video_audit row 'failed'; it never
    raises past this function, so it never affects the linked audio audit.
    """
    from video_analysis import frame_extractor, vision_analyzer

    def _log(event_type: str, detail: dict | None = None):
        activity_log.log_event(event_type, session_id=linked_session_id, associate=label,
                                actor=actor, detail={"video_audit_id": video_audit_id, **(detail or {})})

    screenshots: list[dict] = []
    try:
        video_audit_store.update_status(video_audit_id, "reading_duration")
        metadata = frame_extractor.probe_metadata(video_path)
        duration = metadata.get("duration_seconds", 0.0)
        if duration <= 0:
            raise RuntimeError("Could not read video duration (corrupt or unsupported file)")
        video_audit_store.set_duration(video_audit_id, duration)

        timestamps = frame_extractor.compute_timestamps(duration)
        if not timestamps:
            raise RuntimeError(f"Video too short to snapshot ({duration:.1f}s)")

        # Local-only, no AI: FFmpeg seek+grab + a cheap black-frame check with
        # nearby-timestamp retry (frame_extractor.extract_frame_validated).
        video_audit_store.update_status(video_audit_id, "extracting")
        out_dir = video_audit_store.SCREENSHOT_DIR / video_audit_id
        for point in timestamps:
            filename = f"frame_{point['percentage']:02d}pct.jpg"
            out_path = out_dir / filename
            ok, actual_t = frame_extractor.extract_frame_validated(video_path, point["seconds"], out_path, duration)
            screenshots.append({
                "frame_id": len(screenshots) + 1,
                "percentage": point["percentage"],
                "label": point["label"],
                "timestamp_seconds": actual_t,
                "timestamp": _seconds_to_hhmmss(actual_t),
                "filename": filename if ok else None,
                "extract_failed": not ok,
            })
        _log("video_analysis_extracted", {"frames": len(screenshots)})

        # Preprocessing manifest recorded now — metadata + screenshot refs —
        # entirely local, no AI cost incurred to get to this point.
        video_audit_store.save_preprocessing(video_audit_id, metadata, screenshots)

        # From here on is the existing, already-in-production vision-AI step
        # (opt-in, real API cost) — unchanged from before this refactor.
        video_audit_store.update_status(video_audit_id, "analyzing")
        for shot in screenshots:
            if shot.get("extract_failed"):
                shot["analysis"] = {"label": shot["label"], "error": "Frame extraction failed"}
                continue
            shot_path = out_dir / shot["filename"]
            shot["analysis"] = vision_analyzer.analyze_frame(shot_path, label=shot["label"])

        video_audit_store.update_status(video_audit_id, "summarizing")
        summary = vision_analyzer.summarize_video([s["analysis"] for s in screenshots])

        # Screenshots have now been read for analysis and won't be needed
        # locally again this run — persist them to R2 (if enabled) before
        # they're recorded as complete, so screenshots_json includes each
        # one's r2_key from the start.
        _persist_screenshots_to_r2(video_audit_id, out_dir, screenshots)

        video_audit_store.complete(video_audit_id, screenshots, summary, datetime.utcnow().isoformat())
        _log("video_analysis_completed", {"summary": summary})
        logger.info(f"Video analysis {video_audit_id} complete for session {linked_session_id}")

    except Exception as exc:
        logger.exception(f"Video analysis failed for {video_audit_id} (session {linked_session_id}): {exc}")
        video_audit_store.fail(video_audit_id, str(exc), datetime.utcnow().isoformat())
        _log("video_analysis_failed", {"error": str(exc)[:300]})


def _seconds_to_hhmmss(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def _run_video_snapshot_job(video_audit_id: str, linked_session_id: str, url: str,
                             label: str, actor: str) -> None:
    """
    Standalone / post-hoc trigger only (POST /api/v1/video-audit on an audit
    that already completed its audio audit at an earlier time) — there's no
    concurrent audio-pipeline download to share, so this does its own.
    For a fresh upload with analyze_video=True, see _run_preprocessed_pipeline
    instead, which shares one download between this and the audio pipeline.
    """
    from utils.url_downloader import download_recording

    temp_dir = settings.upload_dir / f"video-{video_audit_id}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        video_audit_store.update_status(video_audit_id, "downloading")
        activity_log.log_event("video_analysis_downloading", session_id=linked_session_id, associate=label,
                                actor=actor, detail={"video_audit_id": video_audit_id})
        video_path = download_recording(url, temp_dir)
        _preprocess_and_analyze_video(video_path, video_audit_id, linked_session_id, label, actor)
    except Exception as exc:
        logger.exception(f"Video analysis download failed for {video_audit_id}: {exc}")
        video_audit_store.fail(video_audit_id, str(exc), datetime.utcnow().isoformat())
        activity_log.log_event("video_analysis_failed", session_id=linked_session_id, associate=label,
                                actor=actor, detail={"video_audit_id": video_audit_id, "error": str(exc)[:300]})
    finally:
        _cleanup_upload_dir(temp_dir)


def _run_preprocessed_pipeline(session_id: str, url: str, upload_dir: Path, tracker,
                                video_audit_id: str, label: str, actor: str) -> None:
    """
    Lightweight video preprocessing, single download shared by both outputs:

        Google Drive URL
              |
        download_recording()            [existing, reused — one call, not two]
              |
        +-----+------------------------------+
        |                                    |
        v                                    v
    5 screenshots + ffprobe metadata     extract_audio_from_local_file()
    (local FFmpeg/Pillow only,           [existing _ffmpeg_extract_mp3,
     no AI, no GPU)                       reused as-is]
        |                                    |
        v                                    v
    (existing, unchanged) vision-AI       _run_pipeline()  [100% EXISTING,
    step on the 5 screenshots             UNCHANGED — transcription,
    — opt-in, unaffected by this          diarization, GPT-4o audit,
    refactor                              scoring, report, history, push]

    Replaces the old pattern of _run_url_pipeline_audio (its own audio-only
    download) running alongside a fully separate _run_video_snapshot_job
    (its own full-video download) — same source file downloaded twice. Only
    used when analyze_video=True; the plain audio-only path is untouched and
    still never downloads the full video at all (see _run_url_pipeline_audio).
    """
    from utils.url_downloader import download_recording, extract_audio_from_local_file

    job_queue.mark_processing(session_id, datetime.utcnow().isoformat())
    video_path = None
    try:
        video_audit_store.update_status(video_audit_id, "downloading")
        tracker.log("Downloading recording once — shared by audio pipeline + screenshots…")
        video_path = download_recording(url, upload_dir)
        _sessions[session_id]["recording_file"] = str(video_path)
        size_mb = video_path.stat().st_size / 1024 / 1024
        tracker.log(f"Download complete: {size_mb:.0f}MB — extracting screenshots and audio…")

        # Local screenshot preprocessing + the existing vision-AI step — same
        # helper the standalone trigger uses, so nothing here is duplicated.
        _preprocess_and_analyze_video(video_path, video_audit_id, session_id, label, actor)

        # Audio extraction for the existing pipeline — reuses the exact same
        # FFmpeg call download_as_audio() already uses elsewhere in this app;
        # no second, duplicate extraction implementation.
        video_audit_store.update_status(video_audit_id, "extracting_audio")
        audio_path = upload_dir / "recording.mp3"
        extract_audio_from_local_file(video_path, audio_path)
        tracker.log("Audio ready — starting existing transcription/audit pipeline…")

        # From here on, 100% the existing, unmodified pipeline.
        _run_pipeline(session_id, audio_path, tracker)

    except Exception as exc:
        logger.exception(f"Preprocessed pipeline failed for {session_id}: {exc}")
        _sessions[session_id].update({"status": "failed", "error": str(exc)})
        _log_audit_failed(session_id, _sessions[session_id], str(exc))
        tracker.error(f"Failed: {str(exc)[:300]}")
        # A failure here (most commonly the shared download itself, which
        # happens before _preprocess_and_analyze_video is even called) would
        # otherwise leave the video_audit row stuck at "queued"/"downloading"
        # forever with no error — _preprocess_and_analyze_video handles its
        # own internal failures, but only once it's actually started.
        rec = video_audit_store.get(video_audit_id)
        if rec and rec["status"] not in ("completed", "failed"):
            video_audit_store.fail(video_audit_id, str(exc), datetime.utcnow().isoformat())
            activity_log.log_event("video_analysis_failed", session_id=session_id, associate=label,
                                    actor=actor, detail={"video_audit_id": video_audit_id, "error": str(exc)[:300]})
        job_queue.mark_failed(session_id, str(exc), datetime.utcnow().isoformat())
    finally:
        # Neither the downloaded video nor any intermediate file is kept —
        # only the small screenshots (already copied out under
        # video_audit_store.SCREENSHOT_DIR) and whatever _run_pipeline itself
        # already produces survive.
        _cleanup_upload_dir(upload_dir)


# ── Audit Records Excel ────────────────────────────────────────────────────────

@app.get("/api/v1/audit/{session_id}/auto-fill")
async def get_autofill(session_id: str):
    """Return auto-fillable fields extracted from the audit report."""
    sess = _sessions.get(session_id)
    if not sess or sess.get("status") != "completed":
        raise HTTPException(404, "Session not found or not completed")
    from reports.audit_excel_manager import auto_fill_from_report, COLUMNS
    session_meta = {
        "source_url": sess.get("source_url", ""),
        "label":      sess.get("label", ""),
        "lead_sheet": sess.get("lead_sheet") or {},
        "video_analysis": video_audit_store.get_by_session(session_id),
    }
    auto = auto_fill_from_report(sess.get("report", {}), session_meta)
    return {"auto_filled": auto, "columns": COLUMNS}


# ── Associate → TL mapping (for auto-filling "TL Name") ───────────────────────

@app.post("/api/v1/tl-mapping/upload")
async def upload_tl_mapping(file: UploadFile = File(...)):
    """Upload a CSV/XLSX with an Associate/Counsellor column and a TL column."""
    from reports.tl_mapping import parse_mapping_file, save_tl_map
    content = await file.read()
    try:
        mapping = parse_mapping_file(content, file.filename or "")
        if not mapping:
            raise ValueError("No rows parsed — check the column headers")
    except Exception as e:
        raise HTTPException(400, f"Could not parse mapping file: {e}")
    save_tl_map(mapping)
    return {"status": "saved", "entries": len(mapping)}


@app.get("/api/v1/tl-mapping")
async def get_tl_mapping():
    from reports.tl_mapping import load_tl_map
    m = load_tl_map()
    return {"entries": len(m), "sample": dict(list(m.items())[:8])}


@app.post("/api/v1/audit/{session_id}/save-to-excel")
async def save_to_excel(session_id: str, row: dict = Body(...), actor: str = ""):
    """Append a completed audit row to the master Excel file."""
    from reports.audit_excel_manager import append_audit_row
    try:
        changes = activity_log.diff_tracker_push(session_id, row)
        row_num = append_audit_row(row)
        activity_log.log_event(
            "pushed_to_tracker", session_id=session_id, associate=row.get("Lead Owner", ""),
            actor=actor, detail={"destination": "excel", "fields": row, "changes": changes, "row": row_num},
        )
        return {"status": "saved", "row": row_num}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/v1/audit-records/download")
async def download_audit_records():
    """Download the master audit records Excel file."""
    from reports.audit_excel_manager import MASTER_PATH, _get_or_create_wb
    _get_or_create_wb()   # ensure file exists
    return FileResponse(
        str(MASTER_PATH),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="kalvium_audit_records.xlsx",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ── WORD MAPPINGS ─────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/v1/word-mappings")
async def get_word_mappings():
    from utils.word_normalizer import load_mappings
    return {"mappings": load_mappings()}


@app.post("/api/v1/word-mappings")
async def add_word_mapping(entry: dict = Body(...)):
    from utils.word_normalizer import load_mappings, save_mappings
    if not entry.get("wrong") or not entry.get("correct"):
        raise HTTPException(400, "Both 'wrong' and 'correct' are required")
    mappings = load_mappings()
    # Check duplicate
    for m in mappings:
        if m["wrong"].lower() == entry["wrong"].lower():
            m["correct"] = entry["correct"]
            m["enabled"] = entry.get("enabled", True)
            save_mappings(mappings)
            return {"status": "updated", "mappings": load_mappings()}
    mappings.append({"wrong": entry["wrong"], "correct": entry["correct"],
                      "enabled": entry.get("enabled", True), "builtin": False})
    save_mappings(mappings)
    return {"status": "added", "mappings": load_mappings()}


@app.patch("/api/v1/word-mappings/{wrong_word}")
async def toggle_word_mapping(wrong_word: str, body: dict = Body(...)):
    from utils.word_normalizer import load_mappings, save_mappings
    mappings = load_mappings()
    for m in mappings:
        if m["wrong"].lower() == wrong_word.lower():
            m["enabled"] = body.get("enabled", True)
            break
    save_mappings(mappings)
    return {"status": "ok", "mappings": load_mappings()}


@app.delete("/api/v1/word-mappings/{wrong_word}")
async def delete_word_mapping(wrong_word: str):
    from utils.word_normalizer import load_mappings, save_mappings
    mappings = [m for m in load_mappings()
                if m["wrong"].lower() != wrong_word.lower() or m.get("builtin")]
    save_mappings(mappings)
    return {"status": "deleted", "mappings": load_mappings()}


# ═══════════════════════════════════════════════════════════════════════════════
# ── SESSION HISTORY ────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/v1/sessions/history")
async def get_history(limit: int = 100, full: bool = False):
    """
    Return list of completed session metadata. full=true (used by the
    history/dashboard carousel) additionally includes each session's
    category scores, strengths, and improvement areas — pulled from the
    stored report_json, one read per session — so the compact card can
    render inline without a second page load per audit.
    """
    rows = session_store.list_sessions(limit)
    sessions = [{
        "session_id":       r["session_id"],
        "date":             r["created_at"],
        "completed_at":     r["completed_at"],
        "label":            r["label"],
        "source_type":      r["source_type"],
        "overall_score":    r["overall_score"],
        "grade":            r["grade"],
        "duration_seconds": r["duration_seconds"],
        "source_url":       r["source_url"],
    } for r in rows]

    if full:
        for s in sessions:
            report = session_store.get_report(s["session_id"]) or {}
            score = report.get("score") or {}
            s["category_scores"] = score.get("category_scores") or {}
            s["top_strengths"] = report.get("top_strengths") or []
            s["improvement_areas"] = report.get("improvement_areas") or []

    return {"sessions": sessions, "total": len(sessions)}


@app.delete("/api/v1/sessions/history/{session_id}")
async def delete_history_entry(session_id: str):
    session_store.delete_session(session_id)
    return {"status": "deleted"}


@app.delete("/api/v1/sessions/history")
async def clear_history():
    session_store.clear_sessions()
    return {"status": "cleared"}


# ═══════════════════════════════════════════════════════════════════════════════
# ── ASSOCIATE INTELLIGENCE ─────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/v1/dashboard/overview")
async def dashboard_overview():
    """All-audits rollup for the Audit Dashboard — KPIs, category breakdown,
    strongest/weakest categories, and trend over time. Read-only aggregation
    over session_store; never touches the audit/scoring pipeline."""
    from reports.associate_analytics import build_overall_dashboard
    return build_overall_dashboard()


@app.get("/api/v1/associates")
async def list_associates():
    """Per-associate rollup: session count, average score, grade mix, trend."""
    from reports.associate_analytics import build_associate_profiles
    return {"associates": build_associate_profiles()}


@app.get("/api/v1/associates/{name}")
async def get_associate(name: str):
    """Single associate's detail — trend + best-effort category breakdown."""
    from reports.associate_analytics import get_associate_detail
    detail = get_associate_detail(name)
    if not detail:
        raise HTTPException(404, "Associate not found")
    return detail


# ═══════════════════════════════════════════════════════════════════════════════
# ── ACTIVITY LOG ────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/v1/activity")
async def get_activity(session_id: str = None, associate: str = None, event_type: str = None,
                        actor: str = None, date_from: str = None, date_to: str = None,
                        search: str = None, limit: int = 200):
    """System-wide activity feed with filters — the audit trail."""
    events = activity_log.list_events(
        session_id=session_id, associate=associate, event_type=event_type, actor=actor,
        date_from=date_from, date_to=date_to, search=search, limit=limit,
    )
    return {"events": events, "total": len(events), "event_types": list(activity_log.EVENT_LABELS.keys())}


@app.get("/api/v1/audit/{session_id}/activity")
async def get_session_activity(session_id: str):
    """Full activity trail for a single audit — its complete history."""
    return {"events": activity_log.list_events(session_id=session_id, limit=500)}


@app.get("/api/v1/audit/{session_id}/lead-sheet")
async def get_session_lead_sheet(session_id: str):
    """
    The full, untouched CRM row this audit was queued with (batch CSV
    upload) — everything, including columns with no home in the tracker
    schema. Nothing from the source file is ever hidden, just not all of
    it necessarily lands in a tracker column.
    """
    s = _sessions.get(session_id)
    if s is not None:
        return {"lead_sheet": s.get("lead_sheet") or {}}
    rec = session_store.get_session(session_id)
    if rec is not None:
        return {"lead_sheet": rec.get("lead_sheet") or {}}
    raise HTTPException(404, "Session not found")


@app.get("/api/v1/audit/{session_id}/tracker-row")
async def get_tracker_row(session_id: str):
    """
    What an edit form should start with. If this audit has ever been
    pushed to the tracker, that's its most recent pushed fields — so
    further edits diff correctly against what's actually sitting in the
    sheet. Otherwise, the same auto-filled row a first push would use.
    """
    from reports.audit_excel_manager import auto_fill_from_report, COLUMNS

    last_push = activity_log.list_events(session_id=session_id, event_type="pushed_to_tracker", limit=1)
    if last_push:
        fields = (last_push[0].get("detail") or {}).get("fields") or {}
        # The pushed snapshot is frozen at push time — if Video Snapshot
        # Analysis finished after that push (a real race: the audio audit is
        # almost always slower, but not guaranteed to be), or ran for the
        # first time afterward via the standalone trigger, its evidence
        # wouldn't be in that old snapshot. Layer today's video fields on
        # top so re-editing always reflects the freshest evidence available,
        # without discarding the rest of the frozen, human-verified push.
        video_rec = video_audit_store.get_by_session(session_id)
        if video_rec and video_rec.get("status") == "completed":
            tf = (video_rec.get("summary") or {}).get("tracker_fields") or {}
            for col in ("Demo Attendees", "Camera Status", "Screen-share Mode (Full screen / PiP)"):
                if tf.get(col):
                    fields[col] = tf[col]
        return {"fields": fields, "columns": COLUMNS, "previously_pushed": True}

    video_rec = video_audit_store.get_by_session(session_id)
    s = _sessions.get(session_id)
    if s is not None:
        report, session_meta = s.get("report") or {}, {
            "source_url": s.get("source_url", ""), "label": s.get("label", ""),
            "lead_sheet": s.get("lead_sheet") or {}, "video_analysis": video_rec,
        }
    else:
        rec = session_store.get_session(session_id)
        if rec is None:
            raise HTTPException(404, "Session not found")
        report, session_meta = rec.get("report") or {}, {
            "source_url": rec.get("source_url", ""), "label": rec.get("label", ""),
            "lead_sheet": rec.get("lead_sheet") or {}, "video_analysis": video_rec,
        }

    fields = auto_fill_from_report(report, session_meta)
    return {"fields": fields, "columns": COLUMNS, "previously_pushed": False}


class VideoAuditRequest(BaseModel):
    session_id: str            # existing demo-audit session to attach this to
    url: Optional[str] = None  # defaults to that session's own source_url if omitted
    actor: Optional[str] = None


@app.post("/api/v1/video-audit")
async def start_video_audit(req: VideoAuditRequest):
    """
    Trigger Video Snapshot & Participant Detection for one existing audit —
    e.g. from the audit's detail page, after the fact, rather than only at
    upload time. Independent job; failure never touches the linked audit.
    """
    s = _sessions.get(req.session_id)
    rec = s or session_store.get_session(req.session_id)
    if rec is None:
        raise HTTPException(404, "Session not found")

    url = (req.url or rec.get("source_url") or "").strip()
    if not url:
        raise HTTPException(400, "No source URL available for this session — pass one explicitly")

    label = rec.get("label", "")
    actor = req.actor or ""
    video_audit_id = str(uuid.uuid4())
    video_audit_store.create(
        video_audit_id, linked_session_id=req.session_id, source_url=url,
        label=label, actor=actor, created_at=datetime.utcnow().isoformat(),
    )
    activity_log.log_event("video_analysis_queued", session_id=req.session_id, associate=label,
                            actor=actor, detail={"video_audit_id": video_audit_id})
    loop = asyncio.get_running_loop()
    loop.run_in_executor(_executor, _run_video_snapshot_job, video_audit_id, req.session_id, url, label, actor)

    return {"video_audit_id": video_audit_id, "status": "queued"}


@app.get("/api/v1/video-audit/by-session/{session_id}")
async def get_video_audit_by_session(session_id: str):
    """
    The most recent Video Snapshot & Participant Detection job attached to
    this demo-audit session, if the auditor opted into it. Returns
    {"video_audit": null} rather than 404 when none was requested — this is
    an optional add-on, so "not run" is a normal, expected state, not an error.
    """
    rec = video_audit_store.get_by_session(session_id)
    return {"video_audit": rec}


@app.get("/api/v1/video-audit/{video_audit_id}")
async def get_video_audit(video_audit_id: str):
    rec = video_audit_store.get(video_audit_id)
    if rec is None:
        raise HTTPException(404, "Video audit not found")
    return rec


SCREENSHOT_PRESIGNED_URL_EXPIRY_SECONDS = 1800  # 30 min — long enough for one UI viewing session


@app.get("/api/v1/video-audit/{video_audit_id}/screenshot/{filename}")
async def get_video_audit_screenshot(video_audit_id: str, filename: str):
    # filename comes only from our own stored screenshots_json (never raw user
    # input), but validate anyway before it touches the filesystem.
    if "/" in filename or ".." in filename:
        raise HTTPException(400, "Invalid filename")

    # If this screenshot was persisted to R2 (STORAGE_BACKEND=r2 at the time
    # it was analyzed), redirect to a short-lived presigned URL rather than
    # exposing a Render filesystem path — the bucket itself stays private.
    rec = video_audit_store.get(video_audit_id)
    if rec:
        shot = next((s for s in rec.get("screenshots", []) if s.get("filename") == filename), None)
        if shot and shot.get("r2_key"):
            from storage.r2 import generate_presigned_download_url, StorageError
            try:
                url = generate_presigned_download_url(
                    shot["r2_key"], expires_in=SCREENSHOT_PRESIGNED_URL_EXPIRY_SECONDS
                )
                return RedirectResponse(url)
            except StorageError as exc:
                logger.error(f"Could not generate presigned URL for screenshot "
                             f"{video_audit_id}/{filename}: {type(exc).__name__}: {exc}")
                raise HTTPException(502, "Could not retrieve screenshot from storage")

    # Local mode (or legacy pre-R2 data with no r2_key) — serve the file
    # directly from local disk, exactly as before this feature existed.
    path = video_audit_store.screenshot_path(video_audit_id, filename)
    if not path.exists():
        raise HTTPException(404, "Screenshot not found")
    return FileResponse(str(path), media_type="image/jpeg")


@app.get("/api/v1/audit/{session_id}/reload")
async def reload_session(session_id: str):
    """Return a stored report for display (from in-memory or the DB)."""
    # First check live sessions
    if session_id in _sessions and _sessions[session_id].get("status") == "completed":
        return _sessions[session_id].get("report", {})
    # Then try session_store (see _add_to_history on completion)
    cached = _load_cached_report(session_id)
    if cached:
        return cached
    raise HTTPException(404, "Session report not found — may have been cleared from memory")


# ═══════════════════════════════════════════════════════════════════════════════
# ── GOOGLE SHEETS INTEGRATION ─────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

# NOTE: this must exactly match an "Authorized redirect URI" on the OAuth
# client in Google Cloud Console. The app is normally served on :8002
# (see CLAUDE.md / start.sh) — override with GOOGLE_REDIRECT_URI in .env
# if you run on a different host/port.
_GOOGLE_REDIRECT = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8002/api/v1/google/callback")


@app.get("/api/v1/google/status")
async def google_status():
    """Return connection status."""
    from reports.google_sheets_manager import has_credentials, has_token, get_config
    cfg = get_config()
    connected = has_token()
    email = ""
    if connected:
        try:
            from reports.google_sheets_manager import get_connected_email
            email = get_connected_email()
        except Exception:
            pass
    return {
        "has_credentials_file": has_credentials(),
        "connected": connected,
        "email": email,
        "sheet_id":   cfg.get("sheet_id", ""),
        "tab_name":   cfg.get("tab_name", "2026 Demo Auditng"),
        "sheet_name": cfg.get("sheet_name", ""),
    }


@app.post("/api/v1/google/upload-credentials")
async def upload_google_credentials(file: UploadFile = File(...)):
    """Accept the credentials.json downloaded from Google Cloud Console."""
    import json as _j
    content = await file.read()
    try:
        data = _j.loads(content)
        # Validate it looks like OAuth credentials
        web_or_installed = data.get("web") or data.get("installed")
        if not web_or_installed:
            raise ValueError("Invalid credentials.json — expected 'web' or 'installed' key")
    except Exception as e:
        raise HTTPException(400, f"Invalid credentials file: {e}")
    from reports.google_sheets_manager import CREDENTIALS_PATH, DATA_DIR, _R2_KEY_CREDENTIALS
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CREDENTIALS_PATH.write_bytes(content)
    from storage.persistent_file import sync_to_r2
    sync_to_r2(CREDENTIALS_PATH, _R2_KEY_CREDENTIALS, content_type="application/json")
    return {"status": "uploaded"}


@app.get("/api/v1/google/auth-url")
async def google_auth_url():
    """Return the Google OAuth consent page URL."""
    from reports.google_sheets_manager import has_credentials, get_auth_url
    if not has_credentials():
        raise HTTPException(400, "Upload credentials.json first")
    url = get_auth_url(_GOOGLE_REDIRECT)
    return {"url": url}


@app.get("/api/v1/google/callback", include_in_schema=False)
async def google_oauth_callback(code: str = "", error: str = ""):
    """OAuth callback — exchange code, store token, close popup."""
    if error:
        html = f"""<html><body><script>
            window.opener && window.opener.postMessage({{type:'google_auth',ok:false,error:{repr(error)}}}, '*');
            window.close();
        </script><p>Auth failed: {error}</p></body></html>"""
        return Response(content=html, media_type="text/html")
    try:
        from reports.google_sheets_manager import exchange_code
        info = exchange_code(code, _GOOGLE_REDIRECT)
        email = info.get("email", "")
        html = f"""<html><body><script>
            window.opener && window.opener.postMessage({{type:'google_auth',ok:true,email:{repr(email)}}}, '*');
            window.close();
        </script><p>Connected as {email}. You can close this window.</p></body></html>"""
        return Response(content=html, media_type="text/html")
    except Exception as e:
        html = f"""<html><body><script>
            window.opener && window.opener.postMessage({{type:'google_auth',ok:false,error:{repr(str(e))}}}, '*');
            window.close();
        </script><p>Error: {e}</p></body></html>"""
        return Response(content=html, media_type="text/html")


@app.post("/api/v1/google/disconnect")
async def google_disconnect():
    from reports.google_sheets_manager import disconnect
    disconnect()
    return {"status": "disconnected"}


@app.get("/api/v1/google/sheets")
async def google_list_sheets():
    """List the user's Google Sheets spreadsheets."""
    from reports.google_sheets_manager import list_spreadsheets
    return {"sheets": list_spreadsheets()}


@app.get("/api/v1/google/sheets/{sheet_id}/tabs")
async def google_sheet_tabs(sheet_id: str):
    from reports.google_sheets_manager import get_sheet_tabs
    return {"tabs": get_sheet_tabs(sheet_id)}


@app.post("/api/v1/google/configure")
async def google_configure(cfg: dict = Body(...)):
    """Save the chosen sheet_id and tab_name."""
    from reports.google_sheets_manager import save_config, ensure_header_row
    save_config(cfg)
    # Optionally write headers
    try:
        ensure_header_row(cfg.get("sheet_id", ""), cfg.get("tab_name", "2026 Demo Auditng"))
    except Exception as e:
        logger.warning(f"Could not write headers: {e}")
    return {"status": "saved"}


@app.post("/api/v1/audit/{session_id}/save-to-gsheet")
async def save_to_gsheet(session_id: str, row: dict = Body(...), actor: str = ""):
    """Append audit row to the configured Google Sheet."""
    from reports.google_sheets_manager import append_row_to_sheet
    try:
        changes = activity_log.diff_tracker_push(session_id, row)
        result = append_row_to_sheet(row)
        activity_log.log_event(
            "pushed_to_tracker", session_id=session_id, associate=row.get("Lead Owner", ""),
            actor=actor, detail={"destination": "gsheet", "fields": row, "changes": changes, **result},
        )
        return {"status": "saved", **result}
    except Exception as e:
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
