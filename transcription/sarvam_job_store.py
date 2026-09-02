"""
transcription/sarvam_job_store.py
Persistent record of every Sarvam Batch STT job — SQLite (data/sarvam_stt_jobs.db)
or Postgres, same thread-local WAL-mode connection pattern as job_queue.py /
session_store.py / video_audit_store.py.

Why this exists: the Batch STT lifecycle (initialise -> upload -> start ->
poll -> download) can span minutes for a 90-120 minute recording, running
inside the same background worker thread that already runs the rest of the
pipeline (see pipeline_v3.py). If the server process restarts mid-job
(deploy, crash), that in-memory thread is gone — but the `sarvam_job_id`
row here survives, so:
  1. we never lose track of which Sarvam job a session paid for, and
  2. a retry (POST /api/v1/queue/{job_id}/retry -> re-run) can check the
     stored job_id's *live* status before deciding whether to submit a new
     job at all — see transcription.sarvam_batch.resolve_existing_job.

One row per audit session (keyed by the same session_id used everywhere
else — job_queue, session_store, activity_log).
"""
from __future__ import annotations
import sqlite3
import threading
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent.parent / "data" / "sarvam_stt_jobs.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# submitted   -> initialise() succeeded, job_id known, upload not yet confirmed
# uploaded    -> upload_files() succeeded
# started     -> start() succeeded, job is Running on Sarvam's side
# polling     -> we are actively waiting (informational only; same as started)
# completed   -> Sarvam job_state == Completed AND we've downloaded+parsed it
# failed      -> Sarvam job_state == Failed
# stt_failed  -> terminal, unrecoverable from our side (max retries, parse
#                error, etc.) — this is the status surfaced to the audit as
#                STT_FAILED per the spec
# interrupted -> orphaned by a server restart while non-terminal (see
#                recover_orphaned()) — a retry re-checks the live Sarvam
#                status rather than assuming the worst
# cancelled   -> user cancelled while we were waiting on this job (see
#                cancel_requested). The Sarvam job itself is NOT killed —
#                cancellation is cooperative (see sarvam_batch.poll_with_backoff)
#                — a later Resume re-checks the job's live status rather
#                than resubmitting.
STATUSES = ["submitted", "uploaded", "started", "polling", "completed",
            "failed", "stt_failed", "interrupted", "cancelled"]

# Maps internal status -> the user-facing stage string shown in the UI,
# matching the requested "Sarvam Batch STT → Job submitted → Processing →
# Completed → Transcript ready" sequence.
STAGE_LABELS: dict[str, str] = {
    "submitted":  "Sarvam Batch STT → Job submitted",
    "uploaded":   "Sarvam Batch STT → Job submitted",
    "started":    "Sarvam Batch STT → Processing",
    "polling":    "Sarvam Batch STT → Processing",
    "completed":  "Sarvam Batch STT → Completed — Transcript ready",
    "failed":     "Sarvam Batch STT → Failed",
    "stt_failed": "Sarvam Batch STT → Failed",
    "interrupted":"Transcription temporarily interrupted",
    "cancelled":  "Transcription cancelled",
}


def stage_label(status: str) -> str:
    return STAGE_LABELS.get(status, status)


_local = threading.local()


def _conn():
    from db.backend import is_postgres_enabled, get_database_url
    if is_postgres_enabled():
        from db.pg import get_pg_connection
        return get_pg_connection(get_database_url(), "sarvam_stt_jobs")
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
    return _local.conn


def init_db() -> None:
    from db.backend import is_postgres_enabled
    if is_postgres_enabled():
        from db.postgres_schema import apply_schema
        apply_schema(_conn(), "sarvam_stt_jobs")
        return
    db = _conn()
    db.executescript("""
    CREATE TABLE IF NOT EXISTS sarvam_stt_jobs (
        session_id       TEXT PRIMARY KEY,
        sarvam_job_id    TEXT DEFAULT '',
        status           TEXT NOT NULL DEFAULT 'submitted',
        wav_path         TEXT DEFAULT '',
        duration_seconds REAL,
        num_segments     INTEGER DEFAULT 1,
        retry_count      INTEGER DEFAULT 0,
        error_message    TEXT DEFAULT '',
        cancel_requested INTEGER NOT NULL DEFAULT 0,
        created_at       TEXT,
        updated_at       TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_sarvam_jobs_status ON sarvam_stt_jobs(status);
    """)
    db.commit()


def upsert_submitted(session_id: str, sarvam_job_id: str, wav_path: str,
                      duration_seconds: float, num_segments: int, now: str) -> None:
    """Called immediately after initialise() succeeds — before upload/start —
    so the job_id is durable even if the process dies on the very next line."""
    db = _conn()
    db.execute("""
        INSERT INTO sarvam_stt_jobs
            (session_id, sarvam_job_id, status, wav_path, duration_seconds,
             num_segments, retry_count, error_message, cancel_requested, created_at, updated_at)
        VALUES (?, ?, 'submitted', ?, ?, ?, 0, '', 0, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
            sarvam_job_id=excluded.sarvam_job_id, status='submitted',
            wav_path=excluded.wav_path, duration_seconds=excluded.duration_seconds,
            num_segments=excluded.num_segments,
            retry_count=sarvam_stt_jobs.retry_count + 1,
            error_message='', cancel_requested=0, updated_at=excluded.updated_at
    """, (session_id, sarvam_job_id, wav_path, duration_seconds, num_segments, now, now))
    db.commit()


def update_status(session_id: str, status: str, error_message: str = "", *, now: str) -> None:
    db = _conn()
    db.execute("""
        UPDATE sarvam_stt_jobs SET status=?, error_message=?, updated_at=?
        WHERE session_id=?
    """, (status, error_message[:1000], now, session_id))
    db.commit()


def request_cancel(session_id: str, now: str) -> bool:
    """Sets the cooperative cancel flag a running poll loop checks between
    polls (see sarvam_batch.poll_with_backoff's is_cancelled callback).
    Does not touch the Sarvam job itself — it keeps running server-side;
    a later Resume checks its live status rather than resubmitting."""
    db = _conn()
    cur = db.execute("""
        UPDATE sarvam_stt_jobs SET cancel_requested=1, updated_at=?
        WHERE session_id=? AND status NOT IN ('completed', 'stt_failed', 'cancelled')
    """, (now, session_id))
    db.commit()
    return cur.rowcount > 0


def is_cancel_requested(session_id: str) -> bool:
    row = _conn().execute(
        "SELECT cancel_requested FROM sarvam_stt_jobs WHERE session_id=?", (session_id,)
    ).fetchone()
    return bool(row and row["cancel_requested"])


def get(session_id: str) -> Optional[dict]:
    row = _conn().execute(
        "SELECT * FROM sarvam_stt_jobs WHERE session_id=?", (session_id,)
    ).fetchone()
    return dict(row) if row else None


def recover_orphaned(now: str) -> int:
    """
    Call once at server startup — mirrors job_queue.recover_orphaned(). Any
    row still in a non-terminal state (submitted/uploaded/started/polling)
    at this point had its worker thread killed by the restart; there is no
    in-memory ProgressTracker or executor task left to finish it. Mark it
    'interrupted' rather than silently leaving it stuck — a retry on that
    session will see 'interrupted' and re-check the stored sarvam_job_id's
    *live* status on Sarvam's side (the job may well have finished there
    even though our poller died) before deciding whether to resubmit.
    Returns the count recovered.
    """
    db = _conn()
    cur = db.execute("""
        UPDATE sarvam_stt_jobs SET status='interrupted', updated_at=?
        WHERE status IN ('submitted', 'uploaded', 'started', 'polling')
    """, (now,))
    db.commit()
    return cur.rowcount
