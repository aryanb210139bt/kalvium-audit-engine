"""
deck/api/routes.py
REST API endpoints for deck management.
"""
from __future__ import annotations
import logging
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

from config.settings import get_settings
from deck import db as deck_db

logger   = logging.getLogger(__name__)
settings = get_settings()
router   = APIRouter(prefix="/api/v1/decks", tags=["decks"])

_executor = ThreadPoolExecutor(max_workers=2)
_ingestion_jobs: dict[str, dict] = {}   # job_id → status


# ── Pydantic models ───────────────────────────────────────────────────────────

class CriterionUpdate(BaseModel):
    importance:          Optional[str]  = None  # must|should|can
    disabled:            Optional[bool] = None
    custom_notes:        Optional[str]  = None
    detection_threshold: Optional[float]= None
    label:               Optional[str]  = None


class WeightsUpdate(BaseModel):
    behavioral_layer:     float = 0.55
    deck_coverage_layer:  float = 0.45


# ── List decks ────────────────────────────────────────────────────────────────

@router.get("")
async def list_decks():
    return {"decks": deck_db.get_all_decks()}


# ── Upload + ingest PPTX ──────────────────────────────────────────────────────

@router.post("/upload")
async def upload_deck(file: UploadFile = File(...), deck_name: str = ""):
    if not file.filename.endswith(".pptx"):
        raise HTTPException(400, "Only .pptx files are supported")

    deck_name = deck_name or Path(file.filename).stem
    upload_dir = Path("uploads/decks")
    upload_dir.mkdir(parents=True, exist_ok=True)

    saved_path = upload_dir / f"{uuid.uuid4().hex}_{file.filename}"
    with open(saved_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    job_id = uuid.uuid4().hex
    _ingestion_jobs[job_id] = {"status": "processing", "deck_name": deck_name}

    def _run():
        try:
            from deck.ingestion.schema_generator import generate_schema_from_pptx
            result = generate_schema_from_pptx(saved_path, deck_name)
            _ingestion_jobs[job_id] = {
                "status": "done",
                "deck_id": result["deck_id"],
                "deck_name": result["deck_name"],
                "slide_count": result["slide_count"],
                "total_criteria": result["total_criteria"],
            }
        except Exception as e:
            logger.exception(f"Ingestion failed for job {job_id}")
            _ingestion_jobs[job_id] = {"status": "failed", "error": str(e)}

    _executor.submit(_run)
    return {"job_id": job_id, "status": "processing",
            "message": f"Ingesting '{deck_name}' — check /api/v1/decks/jobs/{job_id}"}


@router.get("/jobs/{job_id}")
async def get_ingestion_job(job_id: str):
    job = _ingestion_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


# ── Get single deck ───────────────────────────────────────────────────────────

@router.get("/{deck_id}")
async def get_deck(deck_id: str):
    deck = deck_db.get_deck(deck_id)
    if not deck:
        raise HTTPException(404, f"Deck {deck_id!r} not found")
    deck["slides"]   = deck_db.get_deck_slides(deck_id)
    deck["criteria"] = deck_db.get_deck_criteria(deck_id, include_disabled=True)
    return deck


# ── Get criteria only ─────────────────────────────────────────────────────────

@router.get("/{deck_id}/criteria")
async def get_criteria(deck_id: str, include_disabled: bool = False):
    if not deck_db.get_deck(deck_id):
        raise HTTPException(404, "Deck not found")
    criteria = deck_db.get_deck_criteria(deck_id, include_disabled=include_disabled)
    # Group by slide
    by_slide: dict[int, list] = {}
    for c in criteria:
        sn = c["slide_number"]
        by_slide.setdefault(sn, []).append(c)
    return {"deck_id": deck_id, "total": len(criteria), "by_slide": by_slide}


# ── Edit a criterion ──────────────────────────────────────────────────────────

@router.patch("/{deck_id}/criteria/{criterion_id}")
async def update_criterion(deck_id: str, criterion_id: str, body: CriterionUpdate):
    if not deck_db.get_deck(deck_id):
        raise HTTPException(404, "Deck not found")
    updates = {k: v for k, v in body.dict().items() if v is not None}
    if not updates:
        raise HTTPException(400, "No fields to update")
    if "disabled" in updates:
        updates["disabled"] = int(updates["disabled"])
    deck_db.update_criterion(criterion_id, updates)
    return {"updated": criterion_id, "fields": list(updates.keys())}


# ── Add custom criterion ──────────────────────────────────────────────────────

class NewCriterion(BaseModel):
    slide_number: int = 0
    label: str
    explanation: str
    importance: str = "should"
    custom_notes: str = ""


@router.post("/{deck_id}/criteria")
async def add_criterion(deck_id: str, body: NewCriterion):
    if not deck_db.get_deck(deck_id):
        raise HTTPException(404, "Deck not found")

    criterion_id = f"{deck_id}_custom_{uuid.uuid4().hex[:8]}"

    # Embed the explanation
    emb = None
    if settings.openai_api_key:
        try:
            import openai
            client = openai.OpenAI(api_key=settings.openai_api_key)
            resp = client.embeddings.create(
                model="text-embedding-3-small", input=[body.explanation]
            )
            emb = resp.data[0].embedding
        except Exception as e:
            logger.warning(f"Embedding new criterion failed: {e}")

    deck_db.save_criterion({
        "criterion_id":      criterion_id,
        "deck_id":           deck_id,
        "slide_number":      body.slide_number,
        "label":             body.label,
        "explanation":       body.explanation,
        "importance":        body.importance,
        "slide_section":     "custom",
        "embedding":         emb,
        "detection_threshold": 0.72,
        "disabled":          False,
        "custom_notes":      body.custom_notes,
        "display_order":     999,
    })

    # Update criteria count
    conn = deck_db._conn()
    conn.execute(
        "UPDATE deck_schemas SET total_criteria = total_criteria + 1 WHERE deck_id=?",
        (deck_id,)
    )
    conn.commit()

    return {"criterion_id": criterion_id, "status": "created"}


# ── Promote deck status ───────────────────────────────────────────────────────

@router.post("/{deck_id}/promote")
async def promote_deck(deck_id: str):
    deck = deck_db.get_deck(deck_id)
    if not deck:
        raise HTTPException(404, "Deck not found")

    transitions = {
        "draft":       "review",
        "review":      "active",
        "active":      "deprecated",
        "deprecated":  "archived",
    }
    current = deck["status"]
    next_status = transitions.get(current)
    if not next_status:
        raise HTTPException(400, f"Cannot promote from status '{current}'")

    # If promoting to active, deprecate current active deck
    if next_status == "active":
        conn = deck_db._conn()
        conn.execute(
            "UPDATE deck_schemas SET status='deprecated' WHERE status='active' AND deck_id!=?",
            (deck_id,)
        )
        conn.commit()

    deck_db.update_deck_status(deck_id, next_status, datetime.utcnow().isoformat())
    return {"deck_id": deck_id, "old_status": current, "new_status": next_status}


# ── Update scoring weights ────────────────────────────────────────────────────

@router.patch("/{deck_id}/weights")
async def update_weights(deck_id: str, body: WeightsUpdate):
    if not deck_db.get_deck(deck_id):
        raise HTTPException(404, "Deck not found")
    import json
    conn = deck_db._conn()
    conn.execute(
        "UPDATE deck_schemas SET scoring_weights=? WHERE deck_id=?",
        (json.dumps({"behavioral_layer": body.behavioral_layer,
                     "deck_coverage_layer": body.deck_coverage_layer}), deck_id)
    )
    conn.commit()
    return {"updated": deck_id, "weights": body.dict()}


# ── Coverage results for a session ───────────────────────────────────────────

@router.get("/coverage/{session_id}")
async def get_session_coverage(session_id: str):
    from deck import db as deck_db
    # Find which deck this session was evaluated against
    conn = deck_db._conn()
    link = conn.execute(
        "SELECT deck_id FROM deck_call_links WHERE session_id=? LIMIT 1", (session_id,)
    ).fetchone()
    if not link:
        return {"session_id": session_id, "deck_id": None, "results": []}
    deck_id = link["deck_id"]
    results = deck_db.get_coverage_results(session_id, deck_id)
    return {"session_id": session_id, "deck_id": deck_id, "results": results}


# ── Active deck info ──────────────────────────────────────────────────────────

@router.get("/active/info")
async def get_active_deck():
    deck = deck_db.get_active_deck()
    if not deck:
        return {"active": False, "deck": None}
    criteria = deck_db.get_deck_criteria(deck["deck_id"])
    must  = sum(1 for c in criteria if c["importance"] == "must")
    should = sum(1 for c in criteria if c["importance"] == "should")
    return {
        "active": True,
        "deck": {
            **deck,
            "must_count":   must,
            "should_count": should,
            "can_count":    len(criteria) - must - should,
        }
    }
