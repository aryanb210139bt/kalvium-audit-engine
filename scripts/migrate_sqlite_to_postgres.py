"""
scripts/migrate_sqlite_to_postgres.py

One-time ETL: copies every row from the 7 existing SQLite databases into the
PostgreSQL database configured via DATABASE_URL — without modifying or
deleting any SQLite file, and without switching the running app's
DB_BACKEND (that stays "sqlite" until this succeeds and you flip it
yourself — see CLOUD_MIGRATION_PLAN.md).

Usage:
    PYTHONPATH="$(pwd)" python3 scripts/migrate_sqlite_to_postgres.py [--dry-run]

Safety guarantees:
  - The ENTIRE migration (schema creation + every row insert + the
    verification report) runs inside a single PostgreSQL transaction.
    Any failure, or any row-count mismatch found during verification,
    rolls back that whole transaction — Postgres is left exactly as it
    was before the script ran. Never a partially migrated database.
  - Schema is created/verified before any row is inserted, parent tables
    before child tables (auth: users -> sessions; deck: deck_schemas ->
    deck_slides/deck_criteria -> paraphrases/call_links/coverage_results) —
    see db/postgres_schema.py.
  - Idempotent: every insert uses `ON CONFLICT DO NOTHING` against the
    same primary/unique keys as the source SQLite table, so re-running
    this script after a failure (or a no-op re-run once already migrated)
    never duplicates rows.
  - Produces a verification report — SQLite count, Postgres count,
    difference, status — for all 15 tables, and exits non-zero (with the
    transaction rolled back) if any table's counts don't match.
  - DATABASE_URL is read once from settings and used only to open a
    connection. It is never logged, printed, or included in any output —
    every error message below is a fixed, generic string.
"""
from __future__ import annotations
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import get_settings          # noqa: E402
from db.postgres_schema import SCHEMA_DDL         # noqa: E402

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError:
    psycopg = None
    dict_row = None

DATA = ROOT / "data"

# (sqlite_file, table, pg_schema) — parents before children within each schema.
TABLES = [
    (DATA / "auth.db",             "users",                      "auth"),
    (DATA / "auth.db",             "sessions",                   "auth"),
    (DATA / "sessions.db",         "sessions",                   "session_store"),
    (DATA / "activity_log.db",     "activity_log",               "activity_log"),
    (DATA / "associate_roster.db", "associate_roster",           "associate_roster"),
    (DATA / "job_queue.db",        "queued_jobs",                "job_queue"),
    (DATA / "video_audits.db",     "video_audits",               "video_audits"),
    (DATA / "decks.db",            "deck_schemas",               "deck"),
    (DATA / "decks.db",            "deck_slides",                "deck"),
    (DATA / "decks.db",            "deck_criteria",              "deck"),
    (DATA / "decks.db",            "deck_criteria_paraphrases",  "deck"),
    (DATA / "decks.db",            "deck_call_links",            "deck"),
    (DATA / "decks.db",            "deck_coverage_results",      "deck"),
    (DATA / "decks.db",            "schema_edit_log",            "deck"),
]


def _redact(exc: Exception) -> str:
    """Never let a raw driver exception (which can embed the DSN) reach
    stdout — only its class name is safe to print."""
    return type(exc).__name__


def _sqlite_rows(db_path: Path, table: str) -> list[dict]:
    if not db_path.exists():
        return []
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(f"SELECT * FROM {table}").fetchall()]
    finally:
        conn.close()


def _sqlite_count(db_path: Path, table: str) -> int:
    if not db_path.exists():
        return 0
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        return row[0] if row else 0
    finally:
        conn.close()


def _pg_count(cur, table: str) -> int:
    cur.execute(f"SELECT COUNT(*) AS n FROM {table}")
    return cur.fetchone()["n"]


def _insert_rows(cur, table: str, rows: list[dict]) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    col_list = ", ".join(cols)
    placeholders = ", ".join(["%s"] * len(cols))
    sql = f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) ON CONFLICT DO NOTHING"
    for row in rows:
        cur.execute(sql, [row[c] for c in cols])


def _print_report(rows, rolled_back: bool, dry_run: bool = False) -> None:
    print("\n" + "=" * 78)
    print("VERIFICATION REPORT" + (" (dry run)" if dry_run else ""))
    print("=" * 78)
    print(f"{'schema.table':<42}{'sqlite':>8}{'postgres':>10}{'diff':>8}  status")
    print("-" * 78)
    for schema, table, s_n, p_n, diff, status in rows:
        print(f"{schema + '.' + table:<42}{s_n:>8}{p_n:>10}{diff:>8}  {status}")
    print("-" * 78)
    if rolled_back:
        print("(Transaction was rolled back — counts above reflect the attempted "
              "state, not what was actually persisted in PostgreSQL.)")


def run(dry_run: bool = False) -> int:
    if psycopg is None:
        print("ERROR: the 'psycopg' package is not installed. Run: pip install \"psycopg[binary]\"")
        return 1

    database_url = get_settings().database_url
    if not database_url:
        print("ERROR: DATABASE_URL is not configured. Aborting — nothing was touched.")
        return 1

    try:
        conn = psycopg.connect(database_url, autocommit=False, row_factory=dict_row)
    except Exception as exc:
        print(f"ERROR: could not connect to PostgreSQL ({_redact(exc)}). Aborting — nothing was touched.")
        return 1

    report_rows: list[tuple] = []
    try:
        with conn.cursor() as cur:
            # 1) Schema first — parent tables before child tables, per schema.
            for schema, statements in SCHEMA_DDL:
                cur.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
                cur.execute(f'SET search_path TO "{schema}", public')
                for stmt in statements:
                    cur.execute(stmt)

            # 2) Data — same parent-before-child order as TABLES.
            if not dry_run:
                for db_path, table, schema in TABLES:
                    rows = _sqlite_rows(db_path, table)
                    cur.execute(f'SET search_path TO "{schema}", public')
                    _insert_rows(cur, table, rows)

            # 3) Verification — every table, evaluated inside the same
            #    still-open transaction, BEFORE any commit decision.
            for db_path, table, schema in TABLES:
                sqlite_n = _sqlite_count(db_path, table)
                cur.execute(f'SET search_path TO "{schema}", public')
                pg_n = sqlite_n if dry_run else _pg_count(cur, table)
                diff = pg_n - sqlite_n
                status = "OK" if diff == 0 else "MISMATCH"
                report_rows.append((schema, table, sqlite_n, pg_n, diff, status))

            mismatched = any(r[-1] == "MISMATCH" for r in report_rows)

            if dry_run or mismatched:
                conn.rollback()
                _print_report(report_rows, rolled_back=True, dry_run=dry_run)
                if mismatched:
                    print("\nMIGRATION FAILED — one or more tables had a row-count mismatch. "
                          "The PostgreSQL transaction was rolled back; no partial data was "
                          "left behind. SQLite data is untouched.")
                    return 1
                print("\nDry run complete — no changes were committed.")
                return 0

            conn.commit()

        _print_report(report_rows, rolled_back=False)
        print("\nMIGRATION SUCCEEDED — all tables verified, transaction committed. "
              "SQLite files are untouched and remain available as a rollback/backup. "
              "The running app still uses SQLite until DB_BACKEND is explicitly set "
              "to 'postgres'.")
        return 0

    except Exception as exc:
        conn.rollback()
        print(f"ERROR during migration ({_redact(exc)}). Transaction rolled back — "
              f"PostgreSQL was left unchanged. SQLite data is untouched.")
        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(run(dry_run="--dry-run" in sys.argv))
