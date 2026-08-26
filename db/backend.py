"""
db/backend.py
Central switch for which database backend the app uses.

Defaults to SQLite — today's behavior, zero risk. Postgres is only used
when DB_BACKEND is explicitly set to "postgres", which should only be done
after scripts/migrate_sqlite_to_postgres.py has succeeded (its verification
report shows every table matching). See CLOUD_MIGRATION_PLAN.md.

SAFETY: DATABASE_URL must never be logged, printed, committed, or included
in any exception message anywhere in this codebase. Every function here
that touches it either returns it for direct use in a connection call, or
returns a fully generic, constant message — never the value itself, and
never a raw exception's own message (which can embed the DSN).
"""
from __future__ import annotations
from config.settings import get_settings


def db_backend() -> str:
    return (get_settings().db_backend or "sqlite").strip().lower()


def is_postgres_enabled() -> bool:
    return db_backend() == "postgres"


def get_database_url() -> str:
    """Returns the configured DATABASE_URL for opening a connection.
    Callers must never log, print, or surface this value in a response or
    exception — pass it straight to a connection call and nothing else."""
    return get_settings().database_url or ""


def database_configured() -> bool:
    """Safe to log/print/return — reveals only whether a value is set."""
    return bool(get_settings().database_url)
