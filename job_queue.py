"""
job_queue.py
Persistent "uploaded but not yet started" queue — SQLite (data/job_queue.db),
same thread-local WAL-mode connection pattern as session_store.py / deck/db.py
/ video_audit_store.py.

The core distinction this module exists to enforce: UPLOAD creates a row
here with status='queued' and does NOT touch any AI/API/transcription
resource. Only an explicit start (POST /api/v1/queue/{job_id}/start or
/start-all) hands the stored payload to the existing, unmodified run
function — the same one every upload used to call immediately.

`payload_json` holds whatever that run function needs later: a URL, a local
file path already written to disk at upload time, pasted transcript text,
or a CSV row + lead_sheet. Nothing here is re-derived from the AI pipeline —
this module only stores what was true at upload time.
"""
from __future__ import annotations
import json
import sqlite3
import threading
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent / "data" / "job_queue.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# 'starting' exists so a click on Start (single or via start-all) can be
# reflected in the UI the instant it's clicked, before a worker thread has
# actually begun running the job (which is when status flips to
# 'processing' — see mark_processing, called from inside each run function).
STATUSES = ["queued", "starting", "processing", "completed", "failed", "cancelled", "removed"]

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
    CREATE TABLE IF NOT EXISTS queued_jobs (
        job_id       TEXT PRIMARY KEY,   -- same id used as session_id once started
        source_type  TEXT NOT NULL,      -- file | url | transcript | csv_row
        label        TEXT DEFAULT '',    -- associate/display name
        actor        TEXT DEFAULT '',
        filename     TEXT DEFAULT '',    -- display only (original filename / URL / row label)
        duration_hint TEXT DEFAULT '',   -- display only, filled in if known at upload time
        payload_json TEXT NOT NULL,      -- everything the run function needs to start later
        status       TEXT NOT NULL DEFAULT 'queued',
        batch_id     TEXT DEFAULT '',    -- groups multi-link/CSV rows uploaded together
        order_index  INTEGER DEFAULT 0,
        created_at   TEXT,
        started_at   TEXT,
        completed_at TEXT,
        error        TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_queued_jobs_status ON queued_jobs(status);
    CREATE INDEX IF NOT EXISTS idx_queued_jobs_order ON queued_jobs(order_index);
    """)
    db.commit()


def enqueue(job_id: str, source_type: str, label: str, actor: str, payload: dict,
            created_at: str, filename: str = "", duration_hint: str = "", batch_id: str = "") -> None:
    db = _conn()
    max_order = db.execute("SELECT COALESCE(MAX(order_index), 0) AS m FROM queued_jobs").fetchone()["m"]
    db.execute("""
        INSERT INTO queued_jobs (job_id, source_type, label, actor, filename, duration_hint,
                                  payload_json, status, batch_id, order_index, created_at)
        VALUES (?,?,?,?,?,?,?, 'queued', ?, ?, ?)
    """, (job_id, source_type, label, actor, filename, duration_hint,
          json.dumps(payload), batch_id, max_order + 1, created_at))
    db.commit()


def get(job_id: str) -> Optional[dict]:
    row = _conn().execute("SELECT * FROM queued_jobs WHERE job_id=?", (job_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["payload"] = json.loads(d.pop("payload_json")) if d.get("payload_json") else {}
    return d


def list_active(limit: int = 500) -> list[dict]:
    """Everything not yet in a terminal state — for the Upload/Queue and
    Upfront Auditing sections. Ordered so manual reordering (order_index) is
    respected for still-queued items, oldest-first otherwise."""
    rows = _conn().execute("""
        SELECT * FROM queued_jobs
        WHERE status IN ('queued','starting','processing')
        ORDER BY order_index ASC, created_at ASC
        LIMIT ?
    """, (limit,)).fetchall()
    out = []
    for row in rows:
        d = dict(row)
        d["payload"] = json.loads(d.pop("payload_json")) if d.get("payload_json") else {}
        out.append(d)
    return out


def list_queued_only(limit: int = 500) -> list[dict]:
    rows = _conn().execute("""
        SELECT * FROM queued_jobs WHERE status='queued'
        ORDER BY order_index ASC, created_at ASC LIMIT ?
    """, (limit,)).fetchall()
    out = []
    for row in rows:
        d = dict(row)
        d["payload"] = json.loads(d.pop("payload_json")) if d.get("payload_json") else {}
        out.append(d)
    return out


def mark_starting(job_id: str) -> None:
    """Set the instant Start is clicked — before a worker thread has
    necessarily picked the job up yet (it may still be waiting behind other
    jobs in the executor's own queue)."""
    db = _conn()
    db.execute("UPDATE queued_jobs SET status='starting' WHERE job_id=? AND status='queued'", (job_id,))
    db.commit()


def recover_orphaned() -> int:
    """
    Call once at server startup. Any job still marked 'starting' or
    'processing' at that point is provably orphaned — no in-memory
    _sessions entry, ProgressTracker, or executor submission survives a
    process restart (whether from --reload, a crash, or a manual restart),
    so such a job can never actually resume or complete on its own. Without
    this, a restart while jobs were mid-flight leaves them permanently
    stuck: not processing, but also not removable or re-startable (see the
    'queued' guard on remove()/mark_starting flow). Resets them to 'queued'
    so they're immediately visible and startable again. Returns the count
    recovered.
    """
    db = _conn()
    cur = db.execute("""
        UPDATE queued_jobs SET status='queued', started_at=NULL
        WHERE status IN ('starting', 'processing')
    """)
    db.commit()
    return cur.rowcount


def mark_processing(job_id: str, started_at: str) -> None:
    """Called from inside the run function itself, as its first action —
    this is the moment a worker thread actually begins executing the job,
    which is what makes 'Start All' safe: jobs still waiting behind the
    executor's own concurrency limit stay 'starting', not falsely
    'processing', until a worker is actually free for them."""
    db = _conn()
    db.execute("UPDATE queued_jobs SET status='processing', started_at=? WHERE job_id=?",
               (started_at, job_id))
    db.commit()


def mark_completed(job_id: str, completed_at: str) -> None:
    db = _conn()
    # Guarded so a job the user soft-cancelled (see cancel() below) doesn't
    # reappear as 'completed' once the still-running background job finally
    # finishes — cancellation is a UI-visibility decision, not a real stop,
    # so the underlying work keeps going, but its result shouldn't un-hide it.
    db.execute("""
        UPDATE queued_jobs SET status='completed', completed_at=?
        WHERE job_id=? AND status != 'cancelled'
    """, (completed_at, job_id))
    db.commit()


def mark_failed(job_id: str, error: str, completed_at: str) -> None:
    db = _conn()
    db.execute("""
        UPDATE queued_jobs SET status='failed', error=?, completed_at=?
        WHERE job_id=? AND status != 'cancelled'
    """, (error[:1000], completed_at, job_id))
    db.commit()


def cancel(job_id: str, cancelled_at: str) -> bool:
    """
    Soft-cancel a 'starting' or 'processing' job — hides it from the queue/
    processing lists immediately. Does NOT stop the actual background
    pipeline thread (no safe interrupt exists for mid-transcription work
    without touching the protected audit/STT pipeline code), so any
    Sarvam/OpenAI cost already committed is not saved, and the job may still
    complete and land in Audit History/the tracker later — this only stops
    it from cluttering this view. Returns False if the job is already in a
    terminal state (nothing to cancel).
    """
    db = _conn()
    cur = db.execute("""
        UPDATE queued_jobs SET status='cancelled', completed_at=?
        WHERE job_id=? AND status IN ('queued', 'starting', 'processing')
    """, (cancelled_at, job_id))
    db.commit()
    return cur.rowcount > 0


def remove(job_id: str) -> bool:
    """Only removes jobs still in 'queued' (not yet started) — matches the
    spec's explicit rule that processing/completed jobs are never silently
    deleted this way. Returns False (no-op) if the job isn't in that state."""
    db = _conn()
    cur = db.execute("UPDATE queued_jobs SET status='removed' WHERE job_id=? AND status='queued'", (job_id,))
    db.commit()
    return cur.rowcount > 0


def reorder(job_id: str, new_order_index: int) -> None:
    db = _conn()
    db.execute("UPDATE queued_jobs SET order_index=? WHERE job_id=? AND status='queued'",
               (new_order_index, job_id))
    db.commit()


def retry(job_id: str, created_at: str) -> bool:
    """Re-queue a failed job using its original stored payload — no need to
    re-upload. Only valid from 'failed'."""
    db = _conn()
    cur = db.execute("""
        UPDATE queued_jobs SET status='queued', error=NULL, started_at=NULL, completed_at=NULL
        WHERE job_id=? AND status='failed'
    """, (job_id,))
    db.commit()
    return cur.rowcount > 0
