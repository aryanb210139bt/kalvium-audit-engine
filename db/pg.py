"""
db/pg.py
Thin PostgreSQL compatibility layer. Used ONLY when DB_BACKEND=postgres.

Goal: let the 7 existing SQLite store modules (session_store.py,
activity_log.py, auth.py, associate_roster.py, job_queue.py,
video_audit_store.py, deck/db.py) keep every one of their CRUD function
bodies byte-for-byte unchanged — only their `_conn()`/`init_db()` functions
branch here. That's achieved by:

  1. `PgConnCompat` — wraps a psycopg connection so `.execute(sql, params)`,
     `.executemany(sql, seq)`, `.commit()` behave like sqlite3.Connection.
     Rows come back as plain dicts (via psycopg's dict_row factory), so the
     existing `dict(row)` / `row["col"]` call sites keep working unchanged.

  2. `translate_sql()` — rewrites the small, fixed set of SQLite-only syntax
     actually used across those 7 modules (verified by reading every module
     and grepping for OR IGNORE / OR REPLACE / literal '?' before writing
     this): `?` placeholders -> `%s`, `INSERT OR IGNORE` -> `ON CONFLICT DO
     NOTHING`, `INSERT OR REPLACE` -> `ON CONFLICT (<keys>) DO UPDATE SET
     ...` (derived from the statement's own column list, not hand-typed, to
     avoid a transcription mistake). Anything else passes through
     unchanged — plain `INSERT`/`SELECT`/`UPDATE`/`DELETE` with `?` already
     translate correctly with just the placeholder swap.

  3. Each module gets its own Postgres SCHEMA (namespace), named after the
     module, with the connection's search_path pointed at it — see
     db/postgres_schema.py. That's what lets `sessions` exist in both
     session_store and auth without colliding, exactly as it does today
     living in two separate SQLite files.

SAFETY: the DATABASE_URL value is used only to open a connection. It is
never stored on this module, never logged, never printed, and every error
path below raises a fully generic message — never the underlying
exception's own message, which can embed the DSN.
"""
from __future__ import annotations
import re
import threading
from typing import Any

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError:  # only a problem if DB_BACKEND=postgres is actually selected
    psycopg = None
    dict_row = None


class DatabaseConfigError(RuntimeError):
    """Raised when Postgres is selected but misconfigured or unreachable.
    Message is always generic — never includes the connection string or
    any credential."""


# ── SQLite -> Postgres statement translation ────────────────────────────────

_CONFLICT_TARGETS = {
    "deck_schemas": ["deck_id"],
    "deck_slides": ["deck_id", "slide_number"],
    "deck_criteria": ["criterion_id"],
    "deck_coverage_results": ["session_id", "deck_id", "criterion_id"],
}
_OR_REPLACE_RE = re.compile(r"INSERT OR REPLACE INTO\s+(\w+)\s*\(([^)]+)\)", re.IGNORECASE)


def _translate_or_replace(sql: str) -> str:
    m = _OR_REPLACE_RE.search(sql)
    if not m:
        raise ValueError("db/pg.py: could not parse an 'INSERT OR REPLACE' statement for translation")
    table = m.group(1)
    cols = [c.strip() for c in m.group(2).split(",")]
    if table not in _CONFLICT_TARGETS:
        raise ValueError(f"db/pg.py: no registered Postgres conflict target for table {table!r}")
    target_cols = _CONFLICT_TARGETS[table]
    update_cols = [c for c in cols if c not in target_cols]
    set_clause = ", ".join(f"{c}=EXCLUDED.{c}" for c in update_cols)
    sql = sql.replace("INSERT OR REPLACE INTO", "INSERT INTO")
    return sql.rstrip() + f" ON CONFLICT ({', '.join(target_cols)}) DO UPDATE SET {set_clause}"


def translate_sql(sql: str) -> str:
    if "INSERT OR IGNORE INTO" in sql:
        sql = sql.replace("INSERT OR IGNORE INTO", "INSERT INTO").rstrip() + " ON CONFLICT DO NOTHING"
    elif "INSERT OR REPLACE INTO" in sql:
        sql = _translate_or_replace(sql)
    return sql.replace("?", "%s")


# ── sqlite3-shaped wrapper around a psycopg connection ──────────────────────

class _CursorCompat:
    __slots__ = ("_cur",)

    def __init__(self, cur):
        self._cur = cur

    def fetchone(self):
        return self._cur.fetchone()

    def fetchall(self):
        return self._cur.fetchall()

    @property
    def rowcount(self):
        return self._cur.rowcount


class PgConnCompat:
    """Wraps a psycopg connection so the 7 modules' existing
    `db.execute(...)` / `db.executemany(...)` / `db.commit()` call sites
    keep working unchanged when DB_BACKEND=postgres."""

    def __init__(self, conn):
        self._conn = conn

    def execute(self, sql: str, params: Any = ()) -> _CursorCompat:
        cur = self._conn.cursor()
        cur.execute(translate_sql(sql), params)
        return _CursorCompat(cur)

    def executemany(self, sql: str, seq_of_params) -> _CursorCompat:
        cur = self._conn.cursor()
        cur.executemany(translate_sql(sql), list(seq_of_params))
        return _CursorCompat(cur)

    def commit(self) -> None:
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


_local = threading.local()


def get_pg_connection(database_url: str, schema: str) -> PgConnCompat:
    """One psycopg connection per (thread, module-schema) — mirrors the
    existing thread-local-per-module sqlite3 pattern, just all pointed at
    the same Postgres database, isolated by schema/search_path instead of
    by separate files."""
    if psycopg is None:
        raise DatabaseConfigError(
            "DB_BACKEND=postgres but the 'psycopg' package is not installed. "
            "Run: pip install \"psycopg[binary]\""
        )
    if not hasattr(_local, "conns"):
        _local.conns = {}
    if schema not in _local.conns or _local.conns[schema] is None:
        try:
            raw = psycopg.connect(database_url, autocommit=False, row_factory=dict_row)
            with raw.cursor() as cur:
                cur.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
                cur.execute(f'SET search_path TO "{schema}", public')
            raw.commit()
        except Exception:
            raise DatabaseConfigError(
                "Could not connect to the configured PostgreSQL database. "
                "Check that DATABASE_URL is set correctly and the database is reachable."
            ) from None
        _local.conns[schema] = PgConnCompat(raw)
    return _local.conns[schema]


def health_check(database_url: str) -> tuple[bool, str]:
    """Returns (ok, message). `message` is always a short constant string —
    safe to log or return from an API endpoint — never the connection
    string and never a raw driver exception message."""
    if not database_url:
        return False, "DATABASE_URL is not configured"
    if psycopg is None:
        return False, "psycopg package is not installed"
    try:
        with psycopg.connect(database_url, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        return True, "PostgreSQL connection OK"
    except Exception:
        return False, "PostgreSQL connection failed"
