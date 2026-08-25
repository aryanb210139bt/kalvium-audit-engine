"""
api/main.py
FastAPI backend — REST API for the demo audit platform.

Endpoints:
  POST /api/v1/audit/upload    — upload recording, start pipeline
  GET  /api/v1/audit/{id}      — get audit result
  GET  /api/v1/audit/{id}/score — quick score summary
  GET  /api/v1/sessions         — list recent sessions
  GET  /health                  — health check
"""
from __future__ import annotations
import asyncio
import logging
import os
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config.models import DemoAuditReport
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

app = FastAPI(
    title="Demo Audit Platform API",
    description="AI-powered education sales demo intelligence",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store (replace with DB in production)
_sessions: dict[str, dict] = {}
_executor = ThreadPoolExecutor(max_workers=2)

ALLOWED_EXTENSIONS = {".mp4", ".mp3", ".wav", ".m4a", ".ogg", ".webm", ".mov"}


# ── Request / Response models ─────────────────────────────────────────────────

class SessionStatus(BaseModel):
    session_id: str
    status: str         # "queued" | "processing" | "completed" | "failed"
    created_at: str
    completed_at: Optional[str] = None
    recording_file: Optional[str] = None
    error: Optional[str] = None


class ScoreSummary(BaseModel):
    session_id: str
    overall_score: float
    grade: str
    duration_seconds: float
    language_detected: str
    counsellor_talk_pct: float
    total_objections: int
    objection_handling_rate: float
    admission_probability: float
    coaching_highlights: list[str]
    top_strengths: list[str]
    improvement_areas: list[str]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save_upload(file: UploadFile, session_id: str) -> Path:
    upload_dir = settings.upload_dir / session_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(file.filename or "recording.mp4").suffix.lower()
    dest = upload_dir / f"recording{ext}"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return dest


def _run_pipeline(session_id: str, recording_path: Path) -> None:
    """Run the full pipeline synchronously (called in thread pool)."""
    from pipeline import DemoAuditPipeline
    try:
        _sessions[session_id]["status"] = "processing"
        pipeline = DemoAuditPipeline()
        report: DemoAuditReport = pipeline.run(recording_path)
        _sessions[session_id].update({
            "status": "completed",
            "completed_at": datetime.utcnow().isoformat(),
            "report": report.model_dump(),
        })
        logger.info(f"Session {session_id} completed — score: {report.score.overall}/100")
    except Exception as exc:
        logger.exception(f"Pipeline failed for session {session_id}: {exc}")
        _sessions[session_id].update({
            "status": "failed",
            "error": str(exc),
        })


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.post("/api/v1/audit/upload", response_model=SessionStatus)
async def upload_recording(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload a demo recording and start the audit pipeline.
    Returns immediately with a session_id to poll for results.
    """
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {ALLOWED_EXTENSIONS}",
        )

    file_size = 0
    content = await file.read()
    file_size = len(content)
    await file.seek(0)

    if file_size > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({file_size // 1024 // 1024}MB). Max: {settings.max_upload_size_mb}MB",
        )

    session_id = str(uuid.uuid4())
    recording_path = _save_upload(file, session_id)

    _sessions[session_id] = {
        "session_id": session_id,
        "status": "queued",
        "created_at": datetime.utcnow().isoformat(),
        "recording_file": str(recording_path),
    }

    # Run pipeline in background thread (non-blocking)
    loop = asyncio.get_event_loop()
    loop.run_in_executor(_executor, _run_pipeline, session_id, recording_path)

    logger.info(f"New session {session_id} queued — file: {file.filename} ({file_size // 1024}KB)")

    return SessionStatus(**{k: v for k, v in _sessions[session_id].items() if k != "report"})


@app.get("/api/v1/audit/{session_id}", response_model=dict)
async def get_audit_result(session_id: str):
    """Get the full audit report for a completed session."""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    if session["status"] != "completed":
        return {"session_id": session_id, "status": session["status"]}
    return session["report"]


@app.get("/api/v1/audit/{session_id}/status", response_model=SessionStatus)
async def get_session_status(session_id: str):
    """Lightweight polling endpoint — check if pipeline is done."""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return SessionStatus(**{k: v for k, v in session.items() if k != "report"})


@app.get("/api/v1/audit/{session_id}/score", response_model=ScoreSummary)
async def get_score_summary(session_id: str):
    """Quick score card — dashboard-friendly summary."""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    if session["status"] != "completed":
        raise HTTPException(status_code=202, detail=f"Pipeline status: {session['status']}")

    report = session["report"]
    return ScoreSummary(
        session_id=session_id,
        overall_score=report["score"]["overall"],
        grade=report["score"]["grade"],
        duration_seconds=report["duration_seconds"],
        language_detected=report["language_detected"],
        counsellor_talk_pct=report["talk_ratio"]["counsellor_pct"],
        total_objections=report["objections"]["total_objections"],
        objection_handling_rate=report["objections"].get("handling_rate", 0.0),
        admission_probability=report["intent"]["admission_probability"],
        coaching_highlights=report["coaching_highlights"],
        top_strengths=report["top_strengths"],
        improvement_areas=report["improvement_areas"],
    )


@app.get("/api/v1/sessions")
async def list_sessions(limit: int = 20):
    """List recent audit sessions (most recent first)."""
    sessions = sorted(
        _sessions.values(),
        key=lambda s: s.get("created_at", ""),
        reverse=True,
    )[:limit]
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


# ── Dev entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
