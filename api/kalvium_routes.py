"""
api/kalvium_routes.py
FastAPI routes for the Kalvium Booking Authenticity & Fraud Detection Platform.

Endpoints:
  POST /api/kalvium/audit/upload            — upload recording file, start pipeline
  POST /api/kalvium/audit/upload-url        — download from URL, start pipeline
  POST /api/kalvium/audit/upload-csv        — batch from CSV of URLs, start pipelines
  GET  /api/kalvium/audit/{id}              — get full audit report
  GET  /api/kalvium/audit/{id}/status       — polling status
  GET  /api/kalvium/audit/{id}/summary      — fraud summary card
  GET  /api/kalvium/sessions                — list recent sessions
  GET  /api/kalvium/associates              — associate analytics
  GET  /api/kalvium/dashboard               — management dashboard metrics
  POST /api/kalvium/audit/{id}/review       — manager review submission
"""
from __future__ import annotations
import asyncio
import csv
import io
import logging
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, date
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, unquote

import httpx
from fastapi import APIRouter, File, UploadFile, HTTPException, Body
from pydantic import BaseModel

from config.kalvium_models import (
    KalviumAuditReport, FinalClassification, FraudRiskLevel, RecommendedAction
)
from config.settings import get_settings

logger    = logging.getLogger(__name__)
settings  = get_settings()
router    = APIRouter(prefix="/api/kalvium", tags=["Kalvium"])
_executor = ThreadPoolExecutor(max_workers=4)

ALLOWED_EXTENSIONS = {".mp3", ".mp4", ".wav", ".m4a", ".ogg", ".webm", ".mov"}

# In-memory session store — replace with PostgreSQL/Redis in production
_sessions: dict[str, dict] = {}
# In-memory associate analytics accumulator
_associate_stats: dict[str, dict] = {}


# ── Request / Response models ─────────────────────────────────────────────────

class KalviumSessionStatus(BaseModel):
    session_id: str
    status: str
    created_at: str
    completed_at: Optional[str] = None
    associate_id: str = ""
    lead_id: str = ""
    error: Optional[str] = None


class FraudSummaryCard(BaseModel):
    session_id: str
    associate_id: str
    associate_name: str
    prospect_name: str
    call_date: str
    duration_seconds: float
    final_classification: str
    fraud_risk_level: str
    fake_probability: float
    registration_authenticity_score: float
    attendance_probability: float
    compliance_score: float
    engagement_score: float
    email_collected: bool
    email_confirmed: bool
    webinar_explained: bool
    webinar_understood: bool
    qualification_completed: bool
    red_flags: list[str]
    missing_steps: list[str]
    recommended_action: str
    manager_summary: str
    urgency: str
    no_show_risk: float


class ManagerReview(BaseModel):
    reviewer_name: str
    review_decision: str  # CONFIRMED_GENUINE | CONFIRMED_FAKE | NEEDS_MORE_INFO | DISMISSED
    review_notes: str = ""


class UrlUploadRequest(BaseModel):
    recording_url: str
    associate_id: str = ""
    associate_name: str = ""
    lead_id: str = ""
    prospect_name: str = ""


class BatchSessionStatus(BaseModel):
    total: int
    queued: list[str]
    failed_urls: list[str]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ext_from_url(url: str) -> str:
    """Guess file extension from a URL. Falls back to .mp3."""
    path = unquote(urlparse(url).path)
    ext  = Path(path).suffix.lower()
    return ext if ext in ALLOWED_EXTENSIONS else ".mp3"


def _download_url(url: str, dest: Path, timeout: int = 120) -> Path:
    """
    Download a recording from a URL to dest path.
    Follows redirects, streams to disk. Works with TeleCMI signed URLs.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(follow_redirects=True, timeout=timeout) as client:
        with client.stream("GET", url) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_bytes(chunk_size=65536):
                    f.write(chunk)
    return dest


def _save_upload(file: UploadFile, session_id: str) -> Path:
    dest_dir = settings.upload_dir / "kalvium" / session_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    ext  = Path(file.filename or "recording.mp3").suffix.lower()
    dest = dest_dir / f"recording{ext}"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return dest


def _run_kalvium_pipeline(
    session_id: str,
    recording_path: Path,
    associate_id: str,
    associate_name: str,
    lead_id: str,
    prospect_name: str,
) -> None:
    try:
        from pipeline_kalvium import KalviumAuditPipeline
    except Exception as imp_err:
        logger.error(f"[{session_id}] Import error: {imp_err}", exc_info=True)
        _sessions[session_id].update({"status": "failed", "error": f"Import error: {imp_err}"})
        return

    try:
        _sessions[session_id]["status"] = "processing"
        logger.info(f"[{session_id}] Pipeline starting — {recording_path}")
        pipeline = KalviumAuditPipeline()
        report: KalviumAuditReport = pipeline.run(
            recording_path=recording_path,
            associate_id=associate_id,
            associate_name=associate_name,
            lead_id=lead_id,
            prospect_name=prospect_name,
        )
        report_dict = report.model_dump()
        _sessions[session_id].update({
            "status": "completed",
            "completed_at": datetime.utcnow().isoformat(),
            "report": report_dict,
        })
        _update_associate_stats(associate_id, associate_name, report)
        logger.info(
            f"[{session_id}] Completed — {report.final_classification} | "
            f"FP={report.fake_probability:.0%}"
        )
    except Exception as exc:
        logger.exception(f"[{session_id}] Pipeline failed: {exc}")
        _sessions[session_id].update({
            "status": "failed",
            "error": str(exc),
        })


def _update_associate_stats(
    associate_id: str, associate_name: str, report: KalviumAuditReport
) -> None:
    if not associate_id:
        return
    if associate_id not in _associate_stats:
        _associate_stats[associate_id] = {
            "associate_id": associate_id,
            "associate_name": associate_name,
            "total_calls": 0,
            "genuine": 0, "weak": 0, "suspicious": 0, "high_risk": 0,
            "total_compliance": 0, "total_engagement": 0,
            "total_authenticity": 0, "total_attendance": 0,
            "email_collected_count": 0, "email_verified_count": 0,
            "qualified_count": 0, "fraud_flags": 0,
        }
    s = _associate_stats[associate_id]
    s["total_calls"] += 1
    s["associate_name"] = associate_name or s["associate_name"]

    fc = report.final_classification.value.lower().replace(" ", "_")
    s[fc] = s.get(fc, 0) + 1
    if report.fraud_risk_level == FraudRiskLevel.HIGH:
        s["fraud_flags"] += 1

    s["total_compliance"]  += report.compliance_score
    s["total_engagement"]  += report.engagement_score
    s["total_authenticity"] += report.registration_authenticity_score
    s["total_attendance"]   += report.attendance_probability * 100
    if report.email_collected:          s["email_collected_count"] += 1
    if report.email_confirmed:          s["email_verified_count"]  += 1
    if report.qualification_completed:  s["qualified_count"] += 1


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/audit/upload", response_model=KalviumSessionStatus)
async def upload_kalvium_recording(
    file: UploadFile = File(...),
    associate_id: str = "",
    associate_name: str = "",
    lead_id: str = "",
    prospect_name: str = "",
):
    """Upload a sales call recording file and start the 5-layer Kalvium audit pipeline."""
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {ext}")

    content    = await file.read()
    file_size  = len(content)
    await file.seek(0)

    if file_size > settings.max_upload_bytes:
        raise HTTPException(413, f"File too large: {file_size // 1024 // 1024}MB")

    session_id     = str(uuid.uuid4())
    recording_path = _save_upload(file, session_id)

    _sessions[session_id] = {
        "session_id":     session_id,
        "status":         "queued",
        "created_at":     datetime.utcnow().isoformat(),
        "associate_id":   associate_id,
        "associate_name": associate_name,
        "lead_id":        lead_id,
        "prospect_name":  prospect_name,
        "source":         "file",
    }

    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        _executor,
        _run_kalvium_pipeline,
        session_id, recording_path,
        associate_id, associate_name, lead_id, prospect_name,
    )

    logger.info(f"[{session_id}] Queued (file) — {file.filename} ({file_size // 1024}KB)")
    return KalviumSessionStatus(**{
        k: v for k, v in _sessions[session_id].items() if k != "report"
    })


@router.post("/audit/upload-url", response_model=KalviumSessionStatus)
async def upload_from_url(req: UrlUploadRequest):
    """
    Download a recording from a URL (e.g. TeleCMI, S3, Drive) and start the audit pipeline.
    Supports any direct MP3/WAV/M4A link including authenticated signed URLs.
    """
    url = req.recording_url.strip()
    if not url.startswith("http"):
        raise HTTPException(400, "recording_url must be a valid http/https URL")

    session_id = str(uuid.uuid4())
    ext        = _ext_from_url(url)
    dest       = settings.upload_dir / "kalvium" / session_id / f"recording{ext}"

    _sessions[session_id] = {
        "session_id":     session_id,
        "status":         "downloading",
        "created_at":     datetime.utcnow().isoformat(),
        "associate_id":   req.associate_id,
        "associate_name": req.associate_name,
        "lead_id":        req.lead_id,
        "prospect_name":  req.prospect_name,
        "source":         "url",
        "recording_url":  url,
    }

    def _download_then_run():
        try:
            logger.info(f"[{session_id}] Downloading: {url[:80]}")
            _download_url(url, dest)
            file_size = dest.stat().st_size
            logger.info(f"[{session_id}] Downloaded {file_size // 1024}KB → {dest.name}")
            _run_kalvium_pipeline(
                session_id, dest,
                req.associate_id, req.associate_name,
                req.lead_id, req.prospect_name,
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"[{session_id}] URL download failed: {e.response.status_code}")
            _sessions[session_id].update({
                "status": "failed",
                "error":  f"Download failed: HTTP {e.response.status_code}",
            })
        except Exception as exc:
            logger.exception(f"[{session_id}] URL pipeline error: {exc}")
            _sessions[session_id].update({"status": "failed", "error": str(exc)})

    loop = asyncio.get_event_loop()
    loop.run_in_executor(_executor, _download_then_run)

    logger.info(f"[{session_id}] Queued (url) — {url[:60]}…")
    return KalviumSessionStatus(**{
        k: v for k, v in _sessions[session_id].items()
        if k in ("session_id", "status", "created_at", "associate_id", "lead_id")
    })


@router.post("/audit/upload-csv", response_model=BatchSessionStatus)
async def upload_csv_batch(file: UploadFile = File(...)):
    """
    Batch-process a CSV file of recording URLs.

    Required column:  recording_url
    Optional columns: associate_id, associate_name, lead_id, prospect_name

    Example CSV:
      recording_url,associate_id,associate_name,lead_id,prospect_name
      https://rest.telecmi.com/v2/play?...,ASSOC001,Ravi,LEAD001,Ankit Sharma
      https://rest.telecmi.com/v2/play?...,ASSOC002,Priya,LEAD002,Sneha Patel
    """
    content = await file.read()
    text    = content.decode("utf-8-sig")   # strip BOM if present

    try:
        reader = csv.DictReader(io.StringIO(text))
        rows   = list(reader)
    except Exception as exc:
        raise HTTPException(400, f"CSV parse error: {exc}")

    if not rows:
        raise HTTPException(400, "CSV is empty")

    # Normalise headers — case-insensitive, strip spaces
    def _col(row: dict, *candidates: str) -> str:
        for c in candidates:
            for k, v in row.items():
                if k.strip().lower() == c.lower():
                    return (v or "").strip()
        return ""

    queued_ids:  list[str] = []
    failed_urls: list[str] = []

    loop = asyncio.get_event_loop()

    for row in rows:
        url = _col(row, "recording_url", "url", "link", "audio_url", "file_url")
        if not url or not url.startswith("http"):
            failed_urls.append(url or "(empty row)")
            continue

        session_id    = str(uuid.uuid4())
        ext           = _ext_from_url(url)
        dest          = settings.upload_dir / "kalvium" / session_id / f"recording{ext}"
        associate_id  = _col(row, "associate_id", "assoc_id", "agent_id")
        associate_name = _col(row, "associate_name", "assoc_name", "agent_name", "name")
        lead_id       = _col(row, "lead_id", "lead")
        prospect_name = _col(row, "prospect_name", "prospect", "student_name", "customer")

        _sessions[session_id] = {
            "session_id":     session_id,
            "status":         "downloading",
            "created_at":     datetime.utcnow().isoformat(),
            "associate_id":   associate_id,
            "associate_name": associate_name,
            "lead_id":        lead_id,
            "prospect_name":  prospect_name,
            "source":         "csv",
            "recording_url":  url,
        }
        queued_ids.append(session_id)

        # Capture loop vars for closure
        _url, _dest, _sid = url, dest, session_id
        _aid, _aname, _lid, _pname = associate_id, associate_name, lead_id, prospect_name

        def _job(url=_url, dest=_dest, sid=_sid,
                 aid=_aid, aname=_aname, lid=_lid, pname=_pname):
            try:
                _download_url(url, dest)
                _run_kalvium_pipeline(sid, dest, aid, aname, lid, pname)
            except Exception as exc:
                logger.error(f"[{sid}] CSV batch job failed: {exc}")
                _sessions[sid].update({"status": "failed", "error": str(exc)})

        loop.run_in_executor(_executor, _job)

    logger.info(
        f"CSV batch: {len(queued_ids)} queued, {len(failed_urls)} invalid rows"
    )
    return BatchSessionStatus(
        total=len(rows),
        queued=queued_ids,
        failed_urls=failed_urls,
    )


@router.get("/audit/{session_id}", response_model=dict)
async def get_kalvium_audit(session_id: str):
    """Get the full Kalvium audit report for a completed session."""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(404, f"Session '{session_id}' not found")
    if session["status"] != "completed":
        return {"session_id": session_id, "status": session["status"]}
    return session["report"]


@router.get("/audit/{session_id}/status", response_model=KalviumSessionStatus)
async def get_kalvium_status(session_id: str):
    """Lightweight polling endpoint."""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(404, f"Session '{session_id}' not found")
    return KalviumSessionStatus(**{
        k: v for k, v in session.items() if k != "report"
    })


@router.get("/audit/{session_id}/summary", response_model=FraudSummaryCard)
async def get_fraud_summary(session_id: str):
    """Fraud summary card — compact view for management dashboard."""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(404, f"Session '{session_id}' not found")
    if session["status"] != "completed":
        raise HTTPException(202, f"Pipeline status: {session['status']}")

    r  = session["report"]
    ms = r.get("manager_summary", {})
    return FraudSummaryCard(
        session_id                    = session_id,
        associate_id                  = r.get("associate_id", ""),
        associate_name                = r.get("associate_name", ""),
        prospect_name                 = r.get("prospect_name", ""),
        call_date                     = r.get("call_date", ""),
        duration_seconds              = r.get("duration_seconds", 0),
        final_classification          = r.get("final_classification", "HIGH RISK"),
        fraud_risk_level              = r.get("fraud_risk_level", "HIGH"),
        fake_probability              = r.get("fake_probability", 1.0),
        registration_authenticity_score = r.get("registration_authenticity_score", 0),
        attendance_probability        = r.get("attendance_probability", 0),
        compliance_score              = r.get("compliance_score", 0),
        engagement_score              = r.get("engagement_score", 0),
        email_collected               = r.get("email_collected", False),
        email_confirmed               = r.get("email_confirmed", False),
        webinar_explained             = r.get("webinar_explained", False),
        webinar_understood            = r.get("webinar_understood", False),
        qualification_completed       = r.get("qualification_completed", False),
        red_flags                     = r.get("red_flags", [])[:5],
        missing_steps                 = r.get("missing_steps", [])[:5],
        recommended_action            = r.get("recommended_action", "FLAG FOR REVIEW"),
        manager_summary               = ms.get("manager_summary", "") if isinstance(ms, dict) else "",
        urgency                       = r.get("urgency", "ROUTINE"),
        no_show_risk                  = r.get("no_show_risk", 1.0),
    )


@router.post("/audit/{session_id}/review")
async def submit_manager_review(session_id: str, review: ManagerReview):
    """Manager submits their override decision for a flagged booking."""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(404, f"Session '{session_id}' not found")
    if session["status"] != "completed":
        raise HTTPException(400, "Cannot review an incomplete session")

    session["manager_review"] = {
        "reviewer":  review.reviewer_name,
        "decision":  review.review_decision,
        "notes":     review.review_notes,
        "reviewed_at": datetime.utcnow().isoformat(),
    }
    logger.info(
        f"[{session_id}] Manager review: {review.review_decision} by {review.reviewer_name}"
    )
    return {"status": "review_saved", "session_id": session_id}


@router.get("/sessions")
async def list_kalvium_sessions(
    limit: int = 30,
    fraud_only: bool = False,
    associate_id: str = "",
):
    """List recent Kalvium audit sessions with summary cards."""
    sessions = sorted(
        _sessions.values(),
        key=lambda s: s.get("created_at", ""),
        reverse=True,
    )

    results = []
    for s in sessions:
        if not s.get("report"):
            if not fraud_only:
                results.append({
                    "session_id": s["session_id"],
                    "status": s["status"],
                    "created_at": s.get("created_at"),
                })
            continue

        r = s["report"]
        if fraud_only and r.get("fraud_risk_level") != "HIGH":
            continue
        if associate_id and r.get("associate_id") != associate_id:
            continue

        results.append({
            "session_id":              s["session_id"],
            "status":                  s["status"],
            "created_at":              s.get("created_at"),
            "associate_id":            r.get("associate_id", ""),
            "associate_name":          r.get("associate_name", ""),
            "prospect_name":           r.get("prospect_name", ""),
            "final_classification":    r.get("final_classification", ""),
            "fraud_risk_level":        r.get("fraud_risk_level", ""),
            "fake_probability": r.get("fake_probability", 0),
            "registration_authenticity_score": r.get("registration_authenticity_score", 0),
            "compliance_score":        r.get("compliance_score", 0),
            "attendance_probability":  r.get("attendance_probability", 0),
            "urgency":                 r.get("urgency", "ROUTINE"),
            "has_review":              bool(s.get("manager_review")),
        })

    return results[:limit]


@router.get("/associates")
async def get_associate_analytics():
    """Return rolling analytics for all associates."""
    output = []
    for assoc_id, s in _associate_stats.items():
        n = max(1, s["total_calls"])
        fake_pct = round((s.get("high_risk", 0) + s.get("suspicious", 0)) / n * 100, 1)
        output.append({
            "associate_id":         assoc_id,
            "associate_name":       s["associate_name"],
            "total_calls":          n,
            "genuine_count":        s.get("genuine", 0),
            "weak_count":           s.get("weak", 0),
            "suspicious_count":     s.get("suspicious", 0),
            "high_risk_count":      s.get("high_risk", 0),
            "fake_booking_pct":     fake_pct,
            "avg_compliance_score": round(s["total_compliance"] / n, 1),
            "avg_engagement_score": round(s["total_engagement"] / n, 1),
            "avg_authenticity_score": round(s["total_authenticity"] / n, 1),
            "avg_attendance_prediction": round(s["total_attendance"] / n, 1),
            "email_collection_rate": round(s["email_collected_count"] / n * 100, 1),
            "email_verification_rate": round(s["email_verified_count"] / n * 100, 1),
            "qualification_rate":   round(s["qualified_count"] / n * 100, 1),
            "fraud_flags_total":    s["fraud_flags"],
            "risk_level": (
                "HIGH"   if fake_pct > 30 or s["fraud_flags"] >= 3 else
                "MEDIUM" if fake_pct > 15 or s["fraud_flags"] >= 1 else
                "LOW"
            ),
        })
    # Sort by fake_booking_pct descending (most suspicious first)
    return sorted(output, key=lambda x: x["fake_booking_pct"], reverse=True)


@router.get("/dashboard")
async def get_management_dashboard():
    """Aggregate management dashboard metrics."""
    completed = [
        s["report"] for s in _sessions.values()
        if s.get("status") == "completed" and s.get("report")
    ]
    total = len(completed) or 1

    classifications = {
        "GENUINE":        sum(1 for r in completed if r.get("final_classification") == "GENUINE"),
        "LIKELY GENUINE": sum(1 for r in completed if r.get("final_classification") == "LIKELY GENUINE"),
        "WEAK":           sum(1 for r in completed if r.get("final_classification") == "WEAK"),
        "SUSPICIOUS":     sum(1 for r in completed if r.get("final_classification") == "SUSPICIOUS"),
        "HIGH RISK":      sum(1 for r in completed if r.get("final_classification") == "HIGH RISK"),
    }

    fake_pct = round(
        (classifications["SUSPICIOUS"] + classifications["HIGH RISK"]) / total * 100, 1
    )

    avg_compliance    = round(sum(r.get("compliance_score", 0) for r in completed) / total, 1)
    avg_engagement    = round(sum(r.get("engagement_score", 0) for r in completed) / total, 1)
    avg_authenticity  = round(sum(r.get("registration_authenticity_score", 0) for r in completed) / total, 1)
    avg_attendance    = round(sum(r.get("attendance_probability", 0) for r in completed) / total * 100, 1)

    email_collected_pct  = round(sum(1 for r in completed if r.get("email_collected")) / total * 100, 1)
    email_verified_pct   = round(sum(1 for r in completed if r.get("email_confirmed")) / total * 100, 1)
    qualified_pct        = round(sum(1 for r in completed if r.get("qualification_completed")) / total * 100, 1)

    high_risk_sessions = [
        {"session_id": s["session_id"],
         "associate_name": s["report"].get("associate_name", ""),
         "fake_probability": s["report"].get("fake_probability", 0),
         "urgency": s["report"].get("urgency", "ROUTINE")}
        for s in _sessions.values()
        if s.get("status") == "completed"
        and s.get("report", {}).get("fraud_risk_level") == "HIGH"
    ]

    return {
        "total_audits":              total,
        "fake_booking_pct":          fake_pct,
        "classification_breakdown":  classifications,
        "avg_compliance_score":      avg_compliance,
        "avg_engagement_score":      avg_engagement,
        "avg_authenticity_score":    avg_authenticity,
        "avg_attendance_prediction": avg_attendance,
        "email_collection_rate":     email_collected_pct,
        "email_verification_rate":   email_verified_pct,
        "qualification_rate":        qualified_pct,
        "high_risk_sessions":        sorted(
            high_risk_sessions,
            key=lambda x: x["fake_probability"],
            reverse=True
        )[:10],
        "associate_count":           len(_associate_stats),
    }
