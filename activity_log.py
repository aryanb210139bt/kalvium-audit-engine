"""
activity_log.py
Append-only activity trail — a timestamped history of what happened to
each audit and each push to the tracker, independent of session_store's
"latest state" table (which only keeps the current snapshot).

Same SQLite pattern as session_store.py / deck/db.py (thread-local
connection, WAL mode).

What this can and can't see:
  - Every action taken through this app is logged: uploads, audit
    completion/failure, downloads, and tracker pushes (with a diff
    against the previous push to the same session).
  - "Actor" is a lightweight name tag the browser sends (see
    static/audit.html's actor-name prompt) — not a real login, so treat
    it as an honesty-system label, not an audit-grade identity.
  - Edits made by hand directly in the Excel file or Google Sheet
    afterward are invisible to us — we only see what WE pushed.
"""
from __future__ import annotations
import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent / "data" / "activity_log.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

_local = threading.local()


def _conn():
    from db.backend import is_postgres_enabled, get_database_url
    if is_postgres_enabled():
        from db.pg import get_pg_connection
        return get_pg_connection(get_database_url(), "activity_log")
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
    return _local.conn


def init_db() -> None:
    from db.backend import is_postgres_enabled
    if is_postgres_enabled():
        from db.postgres_schema import apply_schema
        apply_schema(_conn(), "activity_log")
        return
    db = _conn()
    db.executescript("""
    CREATE TABLE IF NOT EXISTS activity_log (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id   TEXT,
        associate    TEXT DEFAULT '',
        event_type   TEXT NOT NULL,
        actor        TEXT DEFAULT '',
        timestamp    TEXT NOT NULL,
        detail_json  TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_activity_session   ON activity_log(session_id);
    CREATE INDEX IF NOT EXISTS idx_activity_associate ON activity_log(associate);
    CREATE INDEX IF NOT EXISTS idx_activity_event     ON activity_log(event_type);
    CREATE INDEX IF NOT EXISTS idx_activity_timestamp ON activity_log(timestamp);
    -- Composite: latest_tracker_lead_owner_by_session() filters on
    -- event_type='pushed_to_tracker' AND sorts by timestamp DESC — this
    -- covers both in one index scan (the dashboard calls that function on
    -- every load) instead of relying on just one of the two single-column
    -- indexes above.
    CREATE INDEX IF NOT EXISTS idx_activity_event_timestamp ON activity_log(event_type, timestamp);
    """)
    db.commit()


EVENT_LABELS = {
    "uploaded":              "File/link uploaded",
    "audit_completed":       "Audit completed",
    "audit_failed":          "Audit failed",
    "pdf_downloaded":        "PDF report downloaded",
    "excel_downloaded":      "Excel report downloaded",
    "transcript_downloaded": "Transcript downloaded",
    "pushed_to_tracker":     "Pushed to tracker",
    "deck_uploaded":         "Deck uploaded",
    "deck_promoted":         "Deck status changed",
    "tl_mapping_uploaded":   "TL mapping uploaded",
}


def log_event(event_type: str, session_id: str = None, associate: str = "",
              actor: str = "", detail: Optional[dict] = None) -> None:
    db = _conn()
    db.execute(
        "INSERT INTO activity_log (session_id, associate, event_type, actor, timestamp, detail_json) "
        "VALUES (?,?,?,?,?,?)",
        (session_id, associate or "", event_type, actor or "",
         datetime.utcnow().isoformat(),
         json.dumps(detail) if detail else None)
    )
    db.commit()


def list_events(session_id: str = None, associate: str = None, event_type: str = None,
                 actor: str = None, date_from: str = None, date_to: str = None,
                 search: str = None, limit: int = 200) -> list[dict]:
    q = "SELECT * FROM activity_log WHERE 1=1"
    params: list = []
    if session_id:
        q += " AND session_id=?"; params.append(session_id)
    if associate:
        q += " AND associate=?"; params.append(associate)
    if event_type:
        q += " AND event_type=?"; params.append(event_type)
    if actor:
        q += " AND actor=?"; params.append(actor)
    if date_from:
        q += " AND timestamp>=?"; params.append(date_from)
    if date_to:
        q += " AND timestamp<=?"; params.append(date_to)
    if search:
        q += " AND (associate LIKE ? OR session_id LIKE ? OR actor LIKE ? OR detail_json LIKE ?)"
        like = f"%{search}%"
        params += [like, like, like, like]
    q += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)
    rows = _conn().execute(q, params).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["detail"] = json.loads(d.pop("detail_json")) if d.get("detail_json") else {}
        d["event_label"] = EVENT_LABELS.get(d["event_type"], d["event_type"])
        out.append(d)
    return out


def latest_tracker_lead_owner_by_session() -> dict:
    """
    session_id -> Lead Owner from that session's most recent tracker push.
    This is more authoritative than the upload-time label: it's the
    deliberate, roster-validated name picked at push time (see
    reports/associate_analytics.py, which uses this to decide who an
    audit actually belongs to).
    """
    rows = _conn().execute(
        "SELECT session_id, detail_json FROM activity_log "
        "WHERE event_type='pushed_to_tracker' ORDER BY timestamp DESC"
    ).fetchall()
    result: dict = {}
    for r in rows:
        sid = r["session_id"]
        if not sid or sid in result:
            continue
        try:
            detail = json.loads(r["detail_json"]) if r["detail_json"] else {}
        except Exception:
            continue
        owner = ((detail.get("fields") or {}).get("Lead Owner") or "").strip()
        if owner:
            result[sid] = owner
    return result


def diff_tracker_push(session_id: str, new_fields: dict) -> dict:
    """
    Compare a new tracker push against the most recent previous push for
    this session (if any), returning {field: {"old": ..., "new": ...}}
    for every field that changed. Call this BEFORE logging the new push.
    """
    prev = list_events(session_id=session_id, event_type="pushed_to_tracker", limit=1)
    if not prev:
        return {}
    prev_fields = (prev[0].get("detail") or {}).get("fields", {})
    changes = {}
    for k, new_v in new_fields.items():
        old_v = prev_fields.get(k, "")
        if str(old_v) != str(new_v) and (old_v or new_v):
            changes[k] = {"old": old_v, "new": new_v}
    return changes
