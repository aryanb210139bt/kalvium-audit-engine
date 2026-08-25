"""
video_audit_store.py
SQLite-backed store for Video Snapshot & Participant Detection jobs — the
optional, per-audit add-on. Own DB file (data/video_audits.db), same
thread-local WAL-mode connection pattern as session_store.py / deck/db.py.

A row here is independent of the main `sessions` table — it's created only
when a user opts in to video analysis for a given audit — but carries
`linked_session_id` so audit-detail.html can look it up by the demo-audit's
session_id and render it as a section of that audit.
"""
from __future__ import annotations
import json
import sqlite3
import threading
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent / "data" / "video_audits.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

SCREENSHOT_DIR = Path(__file__).parent / "data" / "video_screenshots"
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

# This job's own lifecycle — separate from (and doesn't touch) the existing
# audio pipeline's own ProgressTracker/WebSocket status, which is untouched.
STATUSES = ["queued", "downloading", "reading_duration", "extracting",
            "extracting_audio", "analyzing", "summarizing", "completed", "failed"]

_local = threading.local()


def _conn() -> sqlite3.Connection:
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
    return _local.conn


def init_db() -> None:
    db = _conn()
    db.executescript("""
    CREATE TABLE IF NOT EXISTS video_audits (
        id                 TEXT PRIMARY KEY,
        linked_session_id  TEXT DEFAULT '',   -- the demo-audit session this belongs to, if any
        source_url         TEXT DEFAULT '',
        label              TEXT DEFAULT '',   -- associate name
        actor              TEXT DEFAULT '',   -- who triggered it
        status             TEXT NOT NULL DEFAULT 'queued',
        duration_seconds   REAL,
        created_at         TEXT,
        completed_at       TEXT,
        error              TEXT,
        screenshots_json   TEXT,              -- list of {label, percentage, seconds, filename, analysis}
        summary_json       TEXT,              -- overall rollup dict
        preprocessing_json TEXT               -- local-only manifest: ffprobe metadata,
                                               -- no AI/API involved — see save_preprocessing()
    );
    CREATE INDEX IF NOT EXISTS idx_video_audits_session ON video_audits(linked_session_id);
    """)
    # Backward-compatible migration for DBs created before preprocessing_json existed.
    cols = {row["name"] for row in db.execute("PRAGMA table_info(video_audits)").fetchall()}
    if "preprocessing_json" not in cols:
        db.execute("ALTER TABLE video_audits ADD COLUMN preprocessing_json TEXT")
    db.commit()


def create(video_audit_id: str, linked_session_id: str, source_url: str,
           label: str, actor: str, created_at: str) -> None:
    db = _conn()
    db.execute("""
        INSERT INTO video_audits (id, linked_session_id, source_url, label, actor,
                                   status, created_at)
        VALUES (?,?,?,?,?, 'queued', ?)
    """, (video_audit_id, linked_session_id, source_url, label, actor, created_at))
    db.commit()


def update_status(video_audit_id: str, status: str, error: Optional[str] = None) -> None:
    db = _conn()
    db.execute("UPDATE video_audits SET status=?, error=? WHERE id=?",
               (status, error, video_audit_id))
    db.commit()


def set_duration(video_audit_id: str, duration_seconds: float) -> None:
    db = _conn()
    db.execute("UPDATE video_audits SET duration_seconds=? WHERE id=?",
               (duration_seconds, video_audit_id))
    db.commit()


def save_preprocessing(video_audit_id: str, metadata: dict, screenshots: list) -> None:
    """
    Record the local-only preprocessing manifest as soon as it's ready — video
    metadata (ffprobe) + screenshot refs+timestamps — independent of whether
    the (optional, separate) vision-AI analysis of those screenshots
    succeeds afterward. No API/AI call happens to produce this data.
    """
    db = _conn()
    manifest = {"metadata": metadata, "screenshots": screenshots}
    db.execute("UPDATE video_audits SET preprocessing_json=? WHERE id=?",
               (json.dumps(manifest), video_audit_id))
    db.commit()


def complete(video_audit_id: str, screenshots: list, summary: dict, completed_at: str) -> None:
    db = _conn()
    db.execute("""
        UPDATE video_audits
        SET status='completed', completed_at=?, screenshots_json=?, summary_json=?
        WHERE id=?
    """, (completed_at, json.dumps(screenshots), json.dumps(summary), video_audit_id))
    db.commit()


def fail(video_audit_id: str, error: str, completed_at: str) -> None:
    db = _conn()
    db.execute("""
        UPDATE video_audits SET status='failed', error=?, completed_at=? WHERE id=?
    """, (error[:500], completed_at, video_audit_id))
    db.commit()


def _row_to_dict(row) -> dict:
    d = dict(row)
    d["screenshots"] = json.loads(d.pop("screenshots_json")) if d.get("screenshots_json") else []
    d["summary"] = json.loads(d.pop("summary_json")) if d.get("summary_json") else {}
    d["preprocessing"] = json.loads(d.pop("preprocessing_json")) if d.get("preprocessing_json") else {}
    return d


def get(video_audit_id: str) -> Optional[dict]:
    row = _conn().execute("SELECT * FROM video_audits WHERE id=?", (video_audit_id,)).fetchone()
    return _row_to_dict(row) if row else None


def get_by_session(linked_session_id: str) -> Optional[dict]:
    """Most recent video-audit job for a given demo-audit session, if any."""
    row = _conn().execute("""
        SELECT * FROM video_audits WHERE linked_session_id=? ORDER BY created_at DESC LIMIT 1
    """, (linked_session_id,)).fetchone()
    return _row_to_dict(row) if row else None


def screenshot_path(video_audit_id: str, filename: str) -> Path:
    return SCREENSHOT_DIR / video_audit_id / filename
