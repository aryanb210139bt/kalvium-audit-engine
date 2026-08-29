"""
tests/conftest.py
Ensures NO test in this suite can ever accidentally operate against a real
Postgres or R2 backend just because the machine's local .env happens to
have DB_BACKEND=postgres / STORAGE_BACKEND=r2 set — as this project's own
local .env legitimately does, for real production use.

Real incident this fixes: a full test-suite run polluted the live Supabase
job_queue and video_audits tables with 44+ fake rows (actor='pytest'),
because test_job_queue.py's and test_video_analysis.py's fixtures only
monkeypatched each module's DB_PATH — which is irrelevant once
is_postgres_enabled() is True, since _conn() checks that first and never
looks at DB_PATH at all. tests/test_storage.py had the same class of bug
for STORAGE_BACKEND/R2 (fixed per-test earlier) — this conftest is the
missing systemic fix: autouse, session-independent, applies to every test
automatically regardless of which file it's in or whether that file
remembers to guard against it.
"""
import pytest


@pytest.fixture(autouse=True)
def _force_local_backends(monkeypatch):
    """Every test runs with DB_BACKEND=sqlite and STORAGE_BACKEND=local, no
    matter what the real environment/.env has configured. Tests must never
    be able to reach a real Postgres or R2 backend by accident."""
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    from config.settings import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
