"""
db/pg.py
Thin PostgreSQL compatibility layer. Used ONLY when DB_BACKEND=postgres.

Goal: let the 7 existing SQLite store modules (session_store.py,
activity_log.py, auth.py, associate_roster.py, job_queue.py,
video_audit_store.py, deck/db.py) keep every one of their CRUD function
bodies byte-for-byte unchanged — only their `_conn()`/`init_db()` functions
branch here. That's achieved by:

  1. `PgConnCompat` — wraps connections checked out from a single shared,
     size-bounded `psycopg_pool.ConnectionPool` so `.execute(sql, params)`,
     `.executemany(sql, seq)`, `.commit()` behave like sqlite3.Connection.
     Rows come back as plain dicts (via psycopg's dict_row factory), so the
     existing `dict(row)` / `row["col"]` call sites keep working unchanged.

     Connection lifetime around the pool: a read-only SELECT checks a
     connection out, runs, and returns it immediately (its result set is
     already fully buffered client-side by the time `.execute()` returns,
     so releasing before `.fetchall()` is safe). A write (anything else)
     holds its connection across however many `.execute()`/`.executemany()`
     calls happen before the next `.commit()` — matching the "batch of
     statements, one commit" pattern already used throughout the 7 modules
     (e.g. deck/db.py's save_coverage_results loop) — and only returns it
     to the pool once that commit lands. Any exception during `.execute()`/
     `.executemany()` rolls the held connection back and returns it to the
     pool immediately, so a failed statement can never leave a cached
     connection stuck in Postgres's "current transaction is aborted" state
     for whatever the next unrelated call happens to be.

     Total physical connections are capped at PG_POOL_MAX_SIZE regardless
     of how many threads or module-schemas are in play — connections are
     shared and reused via the pool rather than pinned one-per-thread
     forever, so the app can't accumulate an unbounded number of them.

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
     module, with every checked-out connection's search_path pointed at it
     on checkout — see db/postgres_schema.py. That's what lets `sessions`
     exist in both session_store and auth without colliding, exactly as it
     does today living in two separate SQLite files.

SAFETY: the DATABASE_URL value is used only to open connections. It is
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

try:
    from psycopg_pool import ConnectionPool
except ImportError:
    ConnectionPool = None

# Total physical connections capped here regardless of thread/schema count —
# comfortably under typical Supabase direct-connection limits, with enough
# headroom for a busy day of concurrent audits.
PG_POOL_MAX_SIZE = 10
PG_POOL_MIN_SIZE = 1
PG_CONNECT_TIMEOUT_SECONDS = 10
PG_CHECKOUT_TIMEOUT_SECONDS = 10


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


def _is_select(translated_sql: str) -> bool:
    return translated_sql.lstrip().upper().startswith("SELECT")


# ── Shared, size-bounded connection pool ────────────────────────────────────
# One pool for the whole process, shared across all 7 module-schemas — safe
# because every checkout resets search_path to whatever schema that
# particular caller needs (see PgConnCompat._checkout), so it doesn't matter
# which schema a given physical connection was last used for.

_pool_lock = threading.Lock()
_pool: "ConnectionPool | None" = None
_pool_database_url: str | None = None


def _get_pool(database_url: str) -> "ConnectionPool":
    global _pool, _pool_database_url
    if psycopg is None:
        raise DatabaseConfigError(
            "DB_BACKEND=postgres but the 'psycopg' package is not installed. "
            "Run: pip install \"psycopg[binary]\""
        )
    if ConnectionPool is None:
        raise DatabaseConfigError(
            "DB_BACKEND=postgres but the 'psycopg_pool' package is not installed. "
            "Run: pip install \"psycopg-pool\""
        )
    if _pool is not None and _pool_database_url == database_url:
        return _pool
    with _pool_lock:
        if _pool is not None and _pool_database_url == database_url:
            return _pool
        if _pool is not None:
            try:
                _pool.close()
            except Exception:
                pass
            _pool = None
        try:
            new_pool = ConnectionPool(
                database_url,
                min_size=PG_POOL_MIN_SIZE,
                max_size=PG_POOL_MAX_SIZE,
                kwargs={
                    "autocommit": False,
                    "row_factory": dict_row,
                    "connect_timeout": PG_CONNECT_TIMEOUT_SECONDS,
                },
                open=True,
            )
        except Exception:
            raise DatabaseConfigError(
                "Could not create the PostgreSQL connection pool. Check that "
                "DATABASE_URL is set correctly and the database is reachable."
            ) from None
        _pool = new_pool
        _pool_database_url = database_url
    return _pool


# ── sqlite3-shaped wrapper around pooled psycopg connections ────────────────

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
    """Wraps checkouts from the shared pool so the 7 modules' existing
    `db.execute(...)` / `db.executemany(...)` / `db.commit()` call sites
    keep working unchanged when DB_BACKEND=postgres. One instance is cached
    per (thread, module-schema) — see get_pg_connection() — but it never
    pins a physical connection for longer than one unit of work (a single
    SELECT, or a batch of writes up to the next commit)."""

    def __init__(self, pool: "ConnectionPool", schema: str):
        self._pool = pool
        self._schema = schema
        self._held_conn = None  # a write-in-progress connection, held until commit()

    def _checkout(self):
        conn = self._pool.getconn(timeout=PG_CHECKOUT_TIMEOUT_SECONDS)
        with conn.cursor() as cur:
            cur.execute(f'SET search_path TO "{self._schema}", public')
        return conn

    def _release(self, conn) -> None:
        """Always rolls back before returning to the pool — the correct
        recovery action whether the connection has a normal open
        transaction (nothing to keep, we're done with it) or is in
        Postgres's aborted-transaction state after a failed statement.
        Never leaves a connection either open-in-transaction or aborted in
        the pool for the next borrower."""
        try:
            conn.rollback()
        except Exception:
            pass  # connection may already be broken — the pool discards bad ones on return
        try:
            self._pool.putconn(conn)
        except Exception:
            pass

    def execute(self, sql: str, params: Any = ()) -> _CursorCompat:
        translated = translate_sql(sql)
        conn = self._held_conn or self._checkout()
        try:
            cur = conn.cursor()
            cur.execute(translated, params)
        except Exception:
            self._held_conn = None
            self._release(conn)
            raise
        if _is_select(translated):
            # Nothing to commit for a read — release now. The cursor's rows
            # are already buffered client-side, so the caller's later
            # fetchone()/fetchall() doesn't need this connection anymore.
            if self._held_conn is None:
                self._release(conn)
        else:
            self._held_conn = conn
        return _CursorCompat(cur)

    def executemany(self, sql: str, seq_of_params) -> _CursorCompat:
        translated = translate_sql(sql)
        conn = self._held_conn or self._checkout()
        try:
            cur = conn.cursor()
            cur.executemany(translated, list(seq_of_params))
        except Exception:
            self._held_conn = None
            self._release(conn)
            raise
        self._held_conn = conn
        return _CursorCompat(cur)

    def commit(self) -> None:
        if self._held_conn is None:
            return  # nothing pending — e.g. called after a read-only execute()
        conn = self._held_conn
        self._held_conn = None
        try:
            conn.commit()
        except Exception:
            self._release(conn)
            raise
        try:
            self._pool.putconn(conn)
        except Exception:
            pass


_local = threading.local()


def get_pg_connection(database_url: str, schema: str) -> PgConnCompat:
    """One PgConnCompat per (thread, module-schema) — mirrors the previous
    thread-local-per-module pattern for callers, but the physical
    connections underneath now come from one shared, bounded pool instead
    of being opened directly and held forever."""
    pool = _get_pool(database_url)
    if not hasattr(_local, "compats"):
        _local.compats = {}
    if schema not in _local.compats:
        _local.compats[schema] = PgConnCompat(pool, schema)
    return _local.compats[schema]


def health_check(database_url: str) -> tuple[bool, str]:
    """Returns (ok, message). `message` is always a short constant string —
    safe to log or return from an API endpoint — never the connection
    string and never a raw driver exception message. Uses its own
    short-lived connection, independent of the shared pool."""
    if not database_url:
        return False, "DATABASE_URL is not configured"
    if psycopg is None:
        return False, "psycopg package is not installed"
    try:
        with psycopg.connect(database_url, connect_timeout=PG_CONNECT_TIMEOUT_SECONDS) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        return True, "PostgreSQL connection OK"
    except Exception:
        return False, "PostgreSQL connection failed"
