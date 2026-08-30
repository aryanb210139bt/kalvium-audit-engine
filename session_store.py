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


def get_reports_bulk(session_ids: list[str]) -> dict[str, dict]:
    """Fetch report_json for MULTIPLE sessions in one query instead of one
    round trip per session — this is what eliminates the N+1 pattern in
    the history 'full=true' response and in the dashboard's category
    aggregation (reports/associate_analytics.py), both of which used to
    call get_report() in a loop. Returns {session_id: parsed_report} for
    whichever of the given ids actually have a report on file (missing/
    empty ones are simply absent from the result, not an error)."""
    if not session_ids:
        return {}
    placeholders = ",".join(["?"] * len(session_ids))
    rows = _conn().execute(
        f"SELECT session_id, report_json FROM sessions WHERE session_id IN ({placeholders})",
        session_ids,
    ).fetchall()
    result: dict[str, dict] = {}
    for r in rows:
        if r["report_json"]:
            result[r["session_id"]] = json.loads(r["report_json"])
    return result


def _search_clause(search: Optional[str]) -> tuple[str, list]:
    """Shared WHERE fragment for list_sessions()/count() — matches label
    (associate name) or session_id, same fields the frontend's search box
    has always searched, just applied server-side now instead of only
    within whatever page happened to already be loaded in the browser.

    LOWER(...) on both sides (rather than bare LIKE) deliberately, not for
    style: SQLite's LIKE is case-insensitive for ASCII by default, but
    Postgres's is case-SENSITIVE — a bare LIKE here would silently behave
    differently depending on DB_BACKEND. Wrapping both sides in LOWER()
    keeps this identical on both, matching the frontend's existing
    .toLowerCase() search."""
    if not search:
        return "", []
    like = f"%{search.lower()}%"
    return " WHERE (LOWER(label) LIKE ? OR LOWER(session_id) LIKE ?)", [like, like]


def _summary_json_exprs() -> str:
    """SQL fragment extracting just the 3 small fields the history page's
    cards actually render (category_scores, top_strengths,
    improvement_areas) directly from the stored report_json TEXT column —
    backend-aware since SQLite and Postgres have different JSON functions.
    Deliberately NOT a full report_json fetch: that column averages ~50KB+
    per row (transcript-level evidence, per-category reasoning text, etc.)
    and get_reports_bulk()/get_report() exist for when the FULL report is
    actually needed (audit detail page) — this is for the history list,
    which only ever displays these three fields per card. See
    list_sessions_page(), which is the only caller."""
    from db.backend import is_postgres_enabled
    if is_postgres_enabled():
        # report_json is stored as TEXT (not jsonb) so both backends can
        # share one column type — cast only for the extraction, and only
        # for the already-LIMIT-ed page (see list_sessions_page's CTE),
        # not the whole table.
        return """,
            (CASE WHEN report_json IS NOT NULL THEN (report_json::jsonb -> 'score' -> 'category_scores') END) AS category_scores_json,
            (CASE WHEN report_json IS NOT NULL THEN (report_json::jsonb -> 'top_strengths') END) AS top_strengths_json,
            (CASE WHEN report_json IS NOT NULL THEN (report_json::jsonb -> 'improvement_areas') END) AS improvement_areas_json"""
    return """,
            json_extract(report_json, '$.score.category_scores') AS category_scores_json,
            json_extract(report_json, '$.top_strengths') AS top_strengths_json,
            json_extract(report_json, '$.improvement_areas') AS improvement_areas_json"""


def list_sessions_page(limit: int = 20, cursor: Optional[tuple] = None,
                        search: Optional[str] = None, include_summary: bool = True,
                        include_total: bool = False) -> tuple[list[dict], Optional[int]]:
    """
    One combined query for the history page: the lightweight row fields,
    the 3 small report-summary fields (see _summary_json_exprs), and
    (optionally) the total matching count — replacing what used to be 3
    sequential round trips (list_sessions + get_reports_bulk + count) with
    1. This is the actual fix for "History stays on Loading for a long
    time": the query execution itself was always sub-millisecond
    (confirmed via EXPLAIN ANALYZE against production — idx_sessions_created_at
    already covers the sort), the cost was almost entirely (a) 3x network
    round-trip latency to a remote Postgres and (b) get_reports_bulk()
    transferring each session's ENTIRE report_json (avg ~50KB+, includes
    full per-category reasoning text) just to read 3 small nested fields.

    cursor: None for the first page. Otherwise (created_at, session_id) —
    the last row of the previous page — for keyset/cursor pagination:
    WHERE (created_at, session_id) < (:cursor_created_at, :cursor_id)
    ORDER BY created_at DESC, session_id DESC LIMIT :limit
    This is what the docstring's business logic is actually doing; kept as
    a plain tuple rather than an opaque token since session_id/created_at
    are already exactly what the frontend has on hand from the last row it
    rendered — no separate encoding scheme needed.

    include_total: only pay for a COUNT on the first page (cursor=None) —
    a "Load more" page reuses the total the client already has, which is
    the second half of eliminating repeated identical-result queries.
    The CTE shape (LIMIT applied to the lightweight columns BEFORE the
    report_json cast/extraction, and total computed via a wholly separate
    subquery) matters, not just cosmetically: confirmed via EXPLAIN ANALYZE
    that folding total into a COUNT(*) OVER() window column instead forces
    Postgres to evaluate the JSON-cast expression for every row in the
    table before the ORDER BY/LIMIT can trim it down to the page size —
    fine at ~60 rows, but it would stop being fine as history grows are the
    exact "loading full audit JSON when only summary fields are needed"
    pattern this whole function exists to avoid, just moved server-side.
    """
    # Built directly (not via _search_clause) so the search condition and
    # the keyset-cursor condition combine with a plain AND regardless of
    # which/how-many of them are present.
    conditions: list[str] = []
    params: list = []
    if search:
        like = f"%{search.lower()}%"
        conditions.append("(LOWER(label) LIKE ? OR LOWER(session_id) LIKE ?)")
        params += [like, like]
    if cursor is not None:
        conditions.append("(created_at, session_id) < (?, ?)")
        params += [cursor[0], cursor[1]]
    where = f" WHERE {' AND '.join(conditions)}" if conditions else ""

    summary_cols = _summary_json_exprs() if include_summary else ""
    page_cols = ("session_id, status, label, source_type, source_url, created_at, "
                 "completed_at, duration_seconds, overall_score, grade" +
                 (", report_json" if include_summary else ""))

    sql = f"""
        WITH page AS (
            SELECT {page_cols}
            FROM sessions{where}
            ORDER BY created_at DESC, session_id DESC
            LIMIT ?
        )
        SELECT session_id, status, label, source_type, source_url, created_at,
               completed_at, duration_seconds, overall_score, grade{summary_cols}
        FROM page
    """
    rows = _conn().execute(sql, params + [limit]).fetchall()
    result = [dict(r) for r in rows]

    total = None
    if include_total:
        # Deliberately a separate, plain COUNT — not COUNT(*) OVER() in the
        # query above, and not scoped to the cursor condition (a "Load
        # more" page's remaining-row count isn't what the UI shows; it
        # shows the one true total from page 1, matching the pre-existing
        # "X of Y audits" display exactly).
        where_no_cursor, params_no_cursor = _search_clause(search)
        total = _conn().execute(
            f"SELECT COUNT(*) AS n FROM sessions{where_no_cursor}", params_no_cursor
        ).fetchone()["n"]

    return result, total


def list_sessions(limit: int = 200, offset: int = 0, search: Optional[str] = None) -> list[dict]:
    """Lightweight summaries (no report_json blob) for history lists and
    associate analytics — newest first. `search` (associate label or
    session id, case-sensitive substring match, mirroring the frontend's
    existing search box) is applied before limit/offset so pagination
    stays correct while searching."""
    where, params = _search_clause(search)
    rows = _conn().execute(f"""
        SELECT session_id, status, label, source_type, source_url, created_at,
               completed_at, duration_seconds, overall_score, grade
        FROM sessions{where} ORDER BY created_at DESC LIMIT ? OFFSET ?
    """, params + [limit, offset]).fetchall()
    return [dict(r) for r in rows]


def delete_session(session_id: str) -> None:
    db = _conn()
    db.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))
    db.commit()


def clear_sessions() -> None:
    db = _conn()
    db.execute("DELETE FROM sessions")
    db.commit()


def count(search: Optional[str] = None) -> int:
    """Total row count — with the same `search` filter as list_sessions()
    when provided, so pagination UI shows the correct total while
    searching (not the unfiltered grand total)."""
    where, params = _search_clause(search)
    return _conn().execute(f"SELECT COUNT(*) AS n FROM sessions{where}", params).fetchone()["n"]


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
