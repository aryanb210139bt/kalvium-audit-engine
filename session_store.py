"""
session_store.py
SQLite-backed store for completed/failed audit sessions — replaces the old
pair of data/session_history.json (summary) + data/reports/{id}.json (full
report) with one indexed table, safe under concurrent writes.

Same connection pattern as deck/db.py (thread-local connection, WAL mode)
so multiple pipeline workers can write at once without clobbering each
other — the JSON-file approach could silently lose a write if two audits
finished in the same second.

In-progress sessions still live in api/main.py's in-memory _sessions dict —
that's legitimately ephemeral, live-progress state. A session only lands
here once it reaches a terminal status (completed or failed).
"""
from __future__ import annotations
import json
import sqlite3
import threading
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent / "data" / "sessions.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

_local = threading.local()


def _conn():
    from db.backend import is_postgres_enabled, get_database_url
    if is_postgres_enabled():
        from db.pg import get_pg_connection
        return get_pg_connection(get_database_url(), "session_store")
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
    return _local.conn


def init_db() -> None:
    from db.backend import is_postgres_enabled
    if is_postgres_enabled():
        from db.postgres_schema import apply_schema
        apply_schema(_conn(), "session_store")
        return
    db = _conn()
    db.executescript("""
    CREATE TABLE IF NOT EXISTS sessions (
        session_id       TEXT PRIMARY KEY,
        status           TEXT NOT NULL,             -- completed | failed
        label            TEXT DEFAULT '',            -- "Lead Owner" / associate name, or URL fallback
        source_type      TEXT DEFAULT '',
        source_url       TEXT DEFAULT '',
        created_at       TEXT,
        completed_at     TEXT,
        duration_seconds REAL,
        overall_score    REAL,
        grade            TEXT,
        error            TEXT,
        lead_sheet_json  TEXT,                       -- phone/source/campaign, JSON blob
        report_json      TEXT                        -- full report dict, JSON blob
    );
    CREATE INDEX IF NOT EXISTS idx_sessions_label      ON sessions(label);
    CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON sessions(created_at);
    """)
    db.commit()


def save_session(session_id: str, sess: dict) -> None:
    """Upsert a terminal-status session. `sess` is the same shape as
    api/main.py's _sessions[session_id] dict."""
    report = sess.get("report") or {}
    score  = report.get("score") or {}
    db = _conn()
    db.execute("""
        INSERT INTO sessions
            (session_id, status, label, source_type, source_url, created_at,
             completed_at, duration_seconds, overall_score, grade, error,
             lead_sheet_json, report_json)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(session_id) DO UPDATE SET
            status=excluded.status, label=excluded.label,
            source_type=excluded.source_type, source_url=excluded.source_url,
            completed_at=excluded.completed_at, duration_seconds=excluded.duration_seconds,
            overall_score=excluded.overall_score, grade=excluded.grade, error=excluded.error,
            lead_sheet_json=excluded.lead_sheet_json, report_json=excluded.report_json
    """, (
        session_id,
        sess.get("status", "completed"),
        sess.get("label", ""),
        sess.get("source_type", ""),
        sess.get("source_url", ""),
        sess.get("created_at", ""),
        sess.get("completed_at", ""),
        report.get("duration_seconds"),
        score.get("overall"),
        score.get("grade"),
        sess.get("error"),
        json.dumps(sess.get("lead_sheet") or {}),
        json.dumps(report) if report else None,
    ))
    db.commit()


def get_session(session_id: str) -> Optional[dict]:
    """Full record including the parsed report — used for GET /api/v1/audit/{id}
    and /reload when the session isn't (or is no longer) in memory."""
    row = _conn().execute("SELECT * FROM sessions WHERE session_id=?", (session_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["report"] = json.loads(d.pop("report_json")) if d.get("report_json") else {}
    d["lead_sheet"] = json.loads(d.pop("lead_sheet_json")) if d.get("lead_sheet_json") else {}
    return d


def get_report(session_id: str) -> Optional[dict]:
    """Just the report blob — the common case (GET /api/v1/audit/{id})."""
    row = _conn().execute("SELECT report_json FROM sessions WHERE session_id=?", (session_id,)).fetchone()
    if not row or not row["report_json"]:
        return None
    return json.loads(row["report_json"])


def list_sessions(limit: int = 200) -> list[dict]:
    """Lightweight summaries (no report_json blob) for history lists and
    associate analytics — newest first."""
    rows = _conn().execute("""
        SELECT session_id, status, label, source_type, source_url, created_at,
               completed_at, duration_seconds, overall_score, grade
        FROM sessions ORDER BY created_at DESC LIMIT ?
    """, (limit,)).fetchall()
    return [dict(r) for r in rows]


def delete_session(session_id: str) -> None:
    db = _conn()
    db.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))
    db.commit()


def clear_sessions() -> None:
    db = _conn()
    db.execute("DELETE FROM sessions")
    db.commit()


def count() -> int:
    return _conn().execute("SELECT COUNT(*) AS n FROM sessions").fetchone()["n"]


# ── One-time migration from the old JSON-file store ────────────────────────────

def migrate_from_json(history_path: Path = Path("data/session_history.json"),
                       reports_dir: Path = Path("data/reports")) -> int:
    """
    Import every entry from the old data/session_history.json (+ matching
    data/reports/{id}.json where present) into the DB. Safe to run more than
    once — existing session_ids are upserted, not duplicated. Returns the
    number of sessions imported.
    """
    if not history_path.exists():
        return 0
    try:
        old_sessions = json.loads(history_path.read_text()).get("sessions", [])
    except Exception:
        return 0

    n = 0
    for s in old_sessions:
        sid = s.get("session_id")
        if not sid:
            continue
        report = {}
        report_path = reports_dir / f"{sid}.json"
        if report_path.exists():
            try:
                report = json.loads(report_path.read_text())
            except Exception:
                report = {}
        elif s.get("overall_score") is not None:
            # No full report cached (predates the report cache) — keep at
            # least the summary score so history/analytics don't lose it.
            report = {"score": {"overall": s.get("overall_score"), "grade": s.get("grade", "")},
                       "duration_seconds": s.get("duration_seconds", 0)}
        save_session(sid, {
            "status": "completed",
            "label": s.get("label", ""),
            "source_type": s.get("source_type", ""),
            "source_url": s.get("source_url", ""),
            "created_at": s.get("date", ""),
            "completed_at": s.get("completed_at", ""),
            "report": report,
        })
        n += 1
    return n
