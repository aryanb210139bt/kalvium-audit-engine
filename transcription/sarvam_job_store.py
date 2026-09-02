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
else — job_queue, session_store, activity_log) for the default case (a
single Sarvam job covers the whole ≤2h recording).

For the rare >2h recording split into N segments (AudioProcessor.
split_for_batch — N always derived dynamically from actual duration, never
a fixed number), each segment gets its OWN row: session_id=
"{parent}::seg{i}", parent_session_id=parent, segment_index=i. This is
what makes segment-level resume correct regardless of N or completion
order:
  - progress is COUNT(status=X) GROUP BY status over the real persisted
    rows for that parent — never "highest completed index" (segments can,
    and with concurrent workers will, finish out of order)
  - a COMPLETED segment's parsed transcript is cached in transcript_json,
    so resume never re-contacts Sarvam for it at all, not even a status
    check — a true zero-cost skip
  - a segment stuck non-terminal past sarvam_batch_stale_threshold_sec
    with no updated_at movement is presumed orphaned (see
    mark_stale_as_interrupted) independently of any one worker crashing,
    not just a full server restart
"""
from __future__ import annotations
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent.parent / "data" / "sarvam_stt_jobs.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# pending     -> row exists (segment plan known) but no Sarvam job submitted yet
# submitted   -> initialise() succeeded, job_id known, upload not yet confirmed
# uploaded    -> upload_files() succeeded
# started     -> start() succeeded, job is Running on Sarvam's side
# polling     -> we are actively waiting (informational only; same as started)
# completed   -> Sarvam job_state == Completed AND transcript_json is cached
# failed      -> Sarvam job_state == Failed
# stt_failed  -> terminal, unrecoverable from our side (max retries, parse
#                error, etc.) — this is the status surfaced to the audit as
#                STT_FAILED per the spec
# interrupted -> orphaned by a server restart (or a stale-worker sweep)
#                while non-terminal — a retry/resume re-checks the stored
#                sarvam_job_id's *live* status rather than assuming the worst
# cancelled   -> user cancelled while we were waiting on this job (see
#                cancel_requested). The Sarvam job itself is NOT killed —
#                cancellation is cooperative (see sarvam_batch.poll_with_backoff)
#                — a later Resume re-checks the job's live status rather
#                than resubmitting.
STATUSES = ["pending", "submitted", "uploaded", "started", "polling", "completed",
            "failed", "stt_failed", "interrupted", "cancelled"]

# Statuses that mean "a worker is (or was) actively working this" — used by
# both the crash-wide recover_orphaned() sweep and the narrower
# per-row mark_stale_as_interrupted() staleness check.
NON_TERMINAL_ACTIVE = ("submitted", "uploaded", "started", "polling")

# Maps internal status -> the user-facing stage string shown in the UI,
# matching the requested "Sarvam Batch STT → Job submitted → Processing →
# Completed → Transcript ready" sequence.
STAGE_LABELS: dict[str, str] = {
    "pending":    "Waiting to start",
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
    # NOTE: the new (parent_session_id/segment_index/transcript_json)
    # columns are deliberately NOT referenced by any CREATE INDEX here yet —
    # on a pre-existing DB (created before these columns existed),
    # CREATE TABLE IF NOT EXISTS is a no-op, so an index on a not-yet-added
    # column would fail immediately. The migration below adds any missing
    # columns first; the column-dependent index is created after that,
    # once the column is guaranteed to exist either way.
    db.executescript("""
    CREATE TABLE IF NOT EXISTS sarvam_stt_jobs (
        session_id        TEXT PRIMARY KEY,
        parent_session_id TEXT DEFAULT '',
        segment_index     INTEGER DEFAULT 0,
        sarvam_job_id     TEXT DEFAULT '',
        status            TEXT NOT NULL DEFAULT 'submitted',
        wav_path          TEXT DEFAULT '',
        duration_seconds  REAL,
        num_segments      INTEGER DEFAULT 1,
        retry_count       INTEGER DEFAULT 0,
        error_message     TEXT DEFAULT '',
        transcript_json   TEXT DEFAULT '',
        cancel_requested  INTEGER NOT NULL DEFAULT 0,
        created_at        TEXT,
        updated_at        TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_sarvam_jobs_status ON sarvam_stt_jobs(status);
    """)
    # Backward-compatible migration for DBs created before these columns
    # existed (same pattern as video_audit_store.init_db()).
    cols = {row["name"] for row in db.execute("PRAGMA table_info(sarvam_stt_jobs)").fetchall()}
    for col, ddl in [
        ("parent_session_id", "ALTER TABLE sarvam_stt_jobs ADD COLUMN parent_session_id TEXT DEFAULT ''"),
        ("segment_index", "ALTER TABLE sarvam_stt_jobs ADD COLUMN segment_index INTEGER DEFAULT 0"),
        ("transcript_json", "ALTER TABLE sarvam_stt_jobs ADD COLUMN transcript_json TEXT DEFAULT ''"),
    ]:
        if col not in cols:
            db.execute(ddl)
    db.execute("CREATE INDEX IF NOT EXISTS idx_sarvam_jobs_parent ON sarvam_stt_jobs(parent_session_id)")
    db.commit()


def upsert_submitted(session_id: str, sarvam_job_id: str, wav_path: str,
                      duration_seconds: float, num_segments: int, now: str,
                      *, parent_session_id: str = "", segment_index: int = 0) -> None:
    """Called immediately after initialise() succeeds — before upload/start —
    so the job_id is durable even if the process dies on the very next line.
    parent_session_id/segment_index are only meaningful the first time a row
    is created (e.g. via create_pending_segments) — on conflict they're left
    untouched, only the job-submission fields advance."""
    db = _conn()
    db.execute("""
        INSERT INTO sarvam_stt_jobs
            (session_id, parent_session_id, segment_index, sarvam_job_id, status,
             wav_path, duration_seconds, num_segments, retry_count, error_message,
             cancel_requested, created_at, updated_at)
        VALUES (?, ?, ?, ?, 'submitted', ?, ?, ?, 0, '', 0, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
            sarvam_job_id=excluded.sarvam_job_id, status='submitted',
            wav_path=excluded.wav_path, duration_seconds=excluded.duration_seconds,
            num_segments=excluded.num_segments,
            retry_count=sarvam_stt_jobs.retry_count + 1,
            error_message='', transcript_json='', cancel_requested=0,
            updated_at=excluded.updated_at
    """, (session_id, parent_session_id, segment_index, sarvam_job_id, wav_path,
          duration_seconds, num_segments, now, now))
    db.commit()


def update_status(session_id: str, status: str, error_message: str = "", *, now: str) -> None:
    db = _conn()
    db.execute("""
        UPDATE sarvam_stt_jobs SET status=?, error_message=?, updated_at=?
        WHERE session_id=?
    """, (status, error_message[:1000], now, session_id))
    db.commit()


def save_transcript(session_id: str, transcript_json: str, *, now: str) -> None:
    """Marks a segment COMPLETED and caches its parsed transcript so a later
    resume/retry never needs to contact Sarvam for this segment again — not
    even a status check. This is what makes "COMPLETED chunk must remain
    reusable" hold at the strongest level, independent of Sarvam job
    retention/expiry."""
    db = _conn()
    db.execute("""
        UPDATE sarvam_stt_jobs SET status='completed', transcript_json=?, updated_at=?
        WHERE session_id=?
    """, (transcript_json, now, session_id))
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


# ── Multi-segment support (>2h recordings — N derived dynamically, never
# hard-coded; see AudioProcessor.split_for_batch) ────────────────────────────

def create_pending_segments(parent_session_id: str, segment_wav_paths: list[str],
                             now: str) -> None:
    """
    Pre-creates one 'pending' row per segment BEFORE any of them are
    processed, so the total N and per-segment identity are durable from the
    very start — a resume/progress query never has to infer N from
    "however many rows happen to exist yet". Idempotent: on a resume where
    rows already exist (in any state), existing rows are left completely
    untouched (ON CONFLICT DO NOTHING) — this must never reset a segment
    that's already progressed back to 'pending'.
    """
    n = len(segment_wav_paths)
    db = _conn()
    for i, wav_path in enumerate(segment_wav_paths):
        seg_session_id = f"{parent_session_id}::seg{i}"
        db.execute("""
            INSERT INTO sarvam_stt_jobs
                (session_id, parent_session_id, segment_index, sarvam_job_id, status,
                 wav_path, duration_seconds, num_segments, retry_count, error_message,
                 cancel_requested, created_at, updated_at)
            VALUES (?, ?, ?, '', 'pending', ?, NULL, ?, 0, '', 0, ?, ?)
            ON CONFLICT(session_id) DO NOTHING
        """, (seg_session_id, parent_session_id, i, wav_path, n, now, now))
    db.commit()


def list_segments(parent_session_id: str) -> list[dict]:
    rows = _conn().execute("""
        SELECT * FROM sarvam_stt_jobs WHERE parent_session_id=? ORDER BY segment_index ASC
    """, (parent_session_id,)).fetchall()
    return [dict(r) for r in rows]


def segment_progress(parent_session_id: str) -> dict:
    """
    Aggregated progress computed from actual persisted per-row statuses —
    COUNT(status=X) GROUP BY status, never "highest segment index seen" —
    correct regardless of N or the order segments actually complete in
    (workers run concurrently; completion order is not guaranteed).
    """
    segments = list_segments(parent_session_id)
    total = len(segments)
    completed = sum(1 for s in segments if s["status"] == "completed")
    failed = sum(1 for s in segments if s["status"] in ("failed", "stt_failed"))
    processing = sum(1 for s in segments if s["status"] in NON_TERMINAL_ACTIVE)
    pending = sum(1 for s in segments if s["status"] in ("pending", "interrupted", "cancelled"))
    return {
        "total_segments": total,
        "completed": completed,
        "processing": processing,
        "pending": pending,
        "failed": failed,
        "progress_pct": round(completed / total * 100, 1) if total else 0.0,
        "segments": [
            {"segment_index": s["segment_index"], "status": s["status"],
             "stage": stage_label(s["status"]), "retry_count": s["retry_count"],
             "error_message": s["error_message"] or None, "updated_at": s["updated_at"]}
            for s in segments
        ],
    }


def recover_orphaned(now: str) -> int:
    """
    Call once at server startup — mirrors job_queue.recover_orphaned(). Any
    row still in a non-terminal state at this point had its worker thread
    killed by the restart; there is no in-memory ProgressTracker or executor
    task left to finish it. Mark it 'interrupted' rather than silently
    leaving it stuck — a retry/resume on that session (or that specific
    segment) will see 'interrupted' and re-check the stored sarvam_job_id's
    *live* status on Sarvam's side (the job may well have finished there
    even though our poller died) before deciding whether to resubmit.
    Returns the count recovered.
    """
    db = _conn()
    placeholders = ",".join("?" * len(NON_TERMINAL_ACTIVE))
    cur = db.execute(f"""
        UPDATE sarvam_stt_jobs SET status='interrupted', updated_at=?
        WHERE status IN ({placeholders})
    """, (now, *NON_TERMINAL_ACTIVE))
    db.commit()
    return cur.rowcount


def mark_stale_as_interrupted(threshold_seconds: int, now: str) -> int:
    """
    Narrower than recover_orphaned(): does NOT assume the whole process
    restarted — sweeps for any individual row whose updated_at (a heartbeat,
    refreshed on every poll — see sarvam_batch.poll_with_backoff's on_status
    callback) is older than threshold_seconds while still non-terminal. This
    catches a single segment's worker dying (thread exception outside the
    normal try/except paths, a hung request past every configured timeout)
    without a full server restart, so a live multi-segment run doesn't get
    stuck on one bad segment forever. Safe to call periodically or on-demand
    (e.g. before computing segment_progress for a "resume" UI check) — a
    genuinely active row's updated_at moves at least once per poll interval
    (<= sarvam_batch_poll_max_sec), well under any sane threshold here.
    """
    db = _conn()
    cutoff = (datetime.fromisoformat(now) - timedelta(seconds=threshold_seconds)).isoformat()
    placeholders = ",".join("?" * len(NON_TERMINAL_ACTIVE))
    cur = db.execute(f"""
        UPDATE sarvam_stt_jobs SET status='interrupted', updated_at=?
        WHERE status IN ({placeholders}) AND updated_at < ?
    """, (now, *NON_TERMINAL_ACTIVE, cutoff))
    db.commit()
    return cur.rowcount
