"""
api/multilingual_routes.py
FastAPI routes for the multilingual call audit pipeline.

Endpoints:
  POST /api/v2/audit/upload      — upload + start multilingual pipeline
  GET  /api/v2/audit/{id}        — get full audit report
  GET  /api/v2/audit/{id}/status — poll pipeline status
  GET  /api/v2/audit/{id}/score  — quick score card
  GET  /api/v2/audit/{id}/coaching — coaching report only
  GET  /api/v2/sessions          — list recent sessions
"""
from __future__ import annotations
import asyncio
import logging
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from config.settings import get_settings

logger   = logging.getLogger(__name__)
settings = get_settings()

router   = APIRouter(prefix="/api/v2", tags=["Multilingual Audit v2"])
_sessions: dict[str, dict] = {}
_executor = ThreadPoolExecutor(max_workers=3)

ALLOWED_EXTENSIONS = {".mp4", ".mp3", ".wav", ".m4a", ".ogg", ".webm", ".mov", ".opus"}


# ── Response models ────────────────────────────────────────────────────────────

class SessionStatus(BaseModel):
    session_id:   str
    status:       str    # queued | processing | completed | failed
    created_at:   str
    completed_at: Optional[str] = None
    error:        Optional[str] = None
    progress_pct: Optional[int] = None


class ScoreCard(BaseModel):
    session_id:         str
    total_score:        float
    grade:              str
    duration_seconds:   float
    primary_language:   str
    deal_risk:          str
    salesperson_pct:    float
    customer_pct:       float
    interruptions:      int
    questions_asked:    int
    pain_points_found:  int
    objections_found:   int
    objections_handled: int
    buying_signals:     int
    next_steps:         int
    top_strengths:      list[str]
    priority_actions:   list[str]
    processing_time_sec: float


# ── Helpers ────────────────────────────────────────────────────────────────────

def _save_upload(file: UploadFile, session_id: str) -> Path:
    upload_dir = settings.upload_dir / "v2" / session_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    ext  = Path(file.filename or "recording.mp4").suffix.lower()
    dest = upload_dir / f"recording{ext}"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return dest


def _run_pipeline(session_id: str, recording_path: Path) -> None:
    from pipeline_multilingual import MultilingualCallAuditPipeline
    try:
        _sessions[session_id]["status"] = "processing"
        _sessions[session_id]["progress_pct"] = 5

        pipeline = MultilingualCallAuditPipeline()
        report = pipeline.run(recording_path)

        _sessions[session_id].update({
            "status":        "completed",
            "completed_at":  datetime.utcnow().isoformat(),
            "report":        report.model_dump(),
            "progress_pct":  100,
        })
        logger.info(
            f"Session {session_id} complete — "
            f"score: {report.sales_audit.total_score}/100 | "
            f"lang: {report.language_map.primary_language.value}"
        )
    except Exception as exc:
        logger.exception(f"Pipeline failed for {session_id}: {exc}")
        _sessions[session_id].update({
            "status": "failed",
            "error":  str(exc),
        })


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.post("/audit/upload", response_model=SessionStatus, status_code=202)
async def upload_recording(file: UploadFile = File(...)):
    """
    Upload a sales call recording and start the multilingual audit pipeline.
    Returns immediately with session_id — poll /status for completion.

    Supports: MP3, MP4, WAV, M4A, OGG, WebM, OPUS, MOV
    Max size: configurable via MAX_UPLOAD_SIZE_MB (default 1GB)
    """
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}"
        )

    content = await file.read()
    size_bytes = len(content)
    await file.seek(0)

    if size_bytes > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_bytes // 1024 // 1024}MB). "
                   f"Max: {settings.max_upload_size_mb}MB"
        )

    session_id     = str(uuid.uuid4())
    recording_path = _save_upload(file, session_id)
    created_at     = datetime.utcnow().isoformat()

    _sessions[session_id] = {
        "session_id":  session_id,
        "status":      "queued",
        "created_at":  created_at,
        "progress_pct": 0,
    }

    loop = asyncio.get_event_loop()
    loop.run_in_executor(_executor, _run_pipeline, session_id, recording_path)

    logger.info(
        f"v2 session {session_id} queued — "
        f"{file.filename} ({size_bytes // 1024}KB)"
    )

    return SessionStatus(
        session_id=session_id,
        status="queued",
        created_at=created_at,
    )


@router.get("/audit/{session_id}/status", response_model=SessionStatus)
async def get_status(session_id: str):
    """Poll pipeline status. Status values: queued | processing | completed | failed."""
    s = _get_session(session_id)
    return SessionStatus(
        session_id=session_id,
        status=s["status"],
        created_at=s.get("created_at", ""),
        completed_at=s.get("completed_at"),
        error=s.get("error"),
        progress_pct=s.get("progress_pct"),
    )


@router.get("/audit/{session_id}", response_model=dict)
async def get_full_report(session_id: str):
    """Get the complete CallAuditReport for a completed session."""
    s = _get_session(session_id)
    if s["status"] != "completed":
        raise HTTPException(
            status_code=202,
            detail=f"Pipeline status: {s['status']}"
        )
    return s["report"]


@router.get("/audit/{session_id}/score", response_model=ScoreCard)
async def get_score_card(session_id: str):
    """Quick score card for dashboards — returns KPIs without full transcript."""
    s = _get_session(session_id)
    if s["status"] != "completed":
        raise HTTPException(status_code=202, detail=f"Status: {s['status']}")

    r = s["report"]
    audit    = r["sales_audit"]
    talk     = r["talk_analytics"]
    intel    = r["call_intelligence"]
    coaching = r["coaching_report"]

    handled = sum(1 for o in intel.get("objections", []) if o.get("handled"))

    return ScoreCard(
        session_id=session_id,
        total_score=audit["total_score"],
        grade=audit["grade"],
        duration_seconds=r["duration_seconds"],
        primary_language=r["language_map"]["primary_language"],
        deal_risk=intel.get("deal_risk", "medium"),
        salesperson_pct=talk["salesperson_pct"],
        customer_pct=talk["customer_pct"],
        interruptions=talk["interruption_count"],
        questions_asked=talk["questions_asked"],
        pain_points_found=len(intel.get("pain_points", [])),
        objections_found=len(intel.get("objections", [])),
        objections_handled=handled,
        buying_signals=len(intel.get("buying_signals", [])),
        next_steps=len(intel.get("next_steps", [])),
        top_strengths=[s["title"] for s in coaching.get("strengths", [])[:3]],
        priority_actions=coaching.get("recommended_actions", [])[:4],
        processing_time_sec=r.get("processing_time_sec", 0),
    )


@router.get("/audit/{session_id}/coaching", response_model=dict)
async def get_coaching_report(session_id: str):
    """Get only the coaching report for a completed session."""
    s = _get_session(session_id)
    if s["status"] != "completed":
        raise HTTPException(status_code=202, detail=f"Status: {s['status']}")
    return s["report"]["coaching_report"]


@router.get("/audit/{session_id}/transcript", response_model=dict)
async def get_transcript(session_id: str):
    """Get the full utterance-level transcript with speaker labels."""
    s = _get_session(session_id)
    if s["status"] != "completed":
        raise HTTPException(status_code=202, detail=f"Status: {s['status']}")
    r = s["report"]
    return {
        "session_id":    session_id,
        "language_map":  r["language_map"],
        "talk_analytics": r["talk_analytics"],
        "utterances":    r["utterances"],
        "call_summary":  r["call_summary"],
    }


@router.get("/sessions", response_model=list)
async def list_sessions(limit: int = 20):
    """List recent audit sessions, most recent first."""
    sorted_sessions = sorted(
        _sessions.values(),
        key=lambda s: s.get("created_at", ""),
        reverse=True,
    )[:limit]

    return [
        {
            "session_id":   s["session_id"],
            "status":       s["status"],
            "created_at":   s.get("created_at"),
            "score":        s["report"]["sales_audit"]["total_score"]
                            if s.get("report") else None,
            "grade":        s["report"]["sales_audit"]["grade"]
                            if s.get("report") else None,
            "language":     s["report"]["language_map"]["primary_language"]
                            if s.get("report") else None,
            "deal_risk":    s["report"]["call_intelligence"]["deal_risk"]
                            if s.get("report") else None,
        }
        for s in sorted_sessions
    ]


def _get_session(session_id: str) -> dict:
    s = _sessions.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return s
