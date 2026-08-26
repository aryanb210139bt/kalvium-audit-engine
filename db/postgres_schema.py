"""
db/postgres_schema.py
PostgreSQL DDL mirroring the 7 SQLite store modules' schemas exactly — same
tables, columns, and constraints, translated only where SQLite/Postgres
syntax genuinely differs:
  - INTEGER PRIMARY KEY AUTOINCREMENT -> INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY
  - REAL -> DOUBLE PRECISION
  - PRAGMA statements dropped (Postgres doesn't need WAL mode / foreign_keys=ON;
    FK enforcement is always on)

Each module gets its OWN Postgres schema (namespace), named after the
module. That's what lets a bare table named `sessions` exist in both
session_store and auth without colliding in the same Postgres database —
each module's connection has its search_path pointed at just its own
schema (+public), exactly mirroring "each module owned its own SQLite
file" — so none of the 7 modules' CRUD call sites need to reference a
schema-qualified name.

Statement order within each schema's list is parent-before-child so
foreign keys always resolve on a fresh database:
  auth:  users -> sessions
  deck:  deck_schemas -> deck_slides / deck_criteria -> paraphrases /
         call_links / coverage_results / schema_edit_log
"""
from __future__ import annotations

SESSION_STORE = [
    """CREATE TABLE IF NOT EXISTS sessions (
        session_id       TEXT PRIMARY KEY,
        status           TEXT NOT NULL,
        label            TEXT DEFAULT '',
        source_type      TEXT DEFAULT '',
        source_url       TEXT DEFAULT '',
        created_at       TEXT,
        completed_at     TEXT,
        duration_seconds DOUBLE PRECISION,
        overall_score    DOUBLE PRECISION,
        grade            TEXT,
        error            TEXT,
        lead_sheet_json  TEXT,
        report_json      TEXT
    )""",
    "CREATE INDEX IF NOT EXISTS idx_sessions_label ON sessions(label)",
    "CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON sessions(created_at)",
]

ACTIVITY_LOG = [
    """CREATE TABLE IF NOT EXISTS activity_log (
        id           INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        session_id   TEXT,
        associate    TEXT DEFAULT '',
        event_type   TEXT NOT NULL,
        actor        TEXT DEFAULT '',
        timestamp    TEXT NOT NULL,
        detail_json  TEXT
    )""",
    "CREATE INDEX IF NOT EXISTS idx_activity_session ON activity_log(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_activity_associate ON activity_log(associate)",
    "CREATE INDEX IF NOT EXISTS idx_activity_event ON activity_log(event_type)",
    "CREATE INDEX IF NOT EXISTS idx_activity_timestamp ON activity_log(timestamp)",
]

AUTH = [
    """CREATE TABLE IF NOT EXISTS users (
        user_id         TEXT PRIMARY KEY,
        email           TEXT UNIQUE NOT NULL,
        name            TEXT NOT NULL,
        password_hash   TEXT NOT NULL,
        password_salt   TEXT NOT NULL,
        role            TEXT NOT NULL DEFAULT 'associate',
        associate_name  TEXT DEFAULT '',
        status          TEXT NOT NULL DEFAULT 'active',
        created_at      TEXT NOT NULL,
        last_login_at   TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS sessions (
        token       TEXT PRIMARY KEY,
        user_id     TEXT NOT NULL REFERENCES users(user_id),
        created_at  TEXT NOT NULL,
        expires_at  TEXT NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id)",
]

ASSOCIATE_ROSTER = [
    """CREATE TABLE IF NOT EXISTS associate_roster (
        name       TEXT PRIMARY KEY,
        added_at   TEXT NOT NULL
    )""",
]

JOB_QUEUE = [
    """CREATE TABLE IF NOT EXISTS queued_jobs (
        job_id        TEXT PRIMARY KEY,
        source_type   TEXT NOT NULL,
        label         TEXT DEFAULT '',
        actor         TEXT DEFAULT '',
        filename      TEXT DEFAULT '',
        duration_hint TEXT DEFAULT '',
        payload_json  TEXT NOT NULL,
        status        TEXT NOT NULL DEFAULT 'queued',
        batch_id      TEXT DEFAULT '',
        order_index   INTEGER DEFAULT 0,
        created_at    TEXT,
        started_at    TEXT,
        completed_at  TEXT,
        error         TEXT
    )""",
    "CREATE INDEX IF NOT EXISTS idx_queued_jobs_status ON queued_jobs(status)",
    "CREATE INDEX IF NOT EXISTS idx_queued_jobs_order ON queued_jobs(order_index)",
]

VIDEO_AUDITS = [
    """CREATE TABLE IF NOT EXISTS video_audits (
        id                 TEXT PRIMARY KEY,
        linked_session_id  TEXT DEFAULT '',
        source_url         TEXT DEFAULT '',
        label              TEXT DEFAULT '',
        actor              TEXT DEFAULT '',
        status             TEXT NOT NULL DEFAULT 'queued',
        duration_seconds   DOUBLE PRECISION,
        created_at         TEXT,
        completed_at       TEXT,
        error              TEXT,
        screenshots_json   TEXT,
        summary_json       TEXT,
        preprocessing_json TEXT
    )""",
    "CREATE INDEX IF NOT EXISTS idx_video_audits_session ON video_audits(linked_session_id)",
]

DECK = [
    """CREATE TABLE IF NOT EXISTS deck_schemas (
        deck_id         TEXT PRIMARY KEY,
        deck_slug       TEXT NOT NULL,
        version_major   INTEGER NOT NULL DEFAULT 1,
        schema_patch    INTEGER NOT NULL DEFAULT 0,
        deck_name       TEXT NOT NULL,
        status          TEXT NOT NULL DEFAULT 'draft',
        scoring_weights TEXT NOT NULL DEFAULT '{"behavioral_layer":0.55,"deck_coverage_layer":0.45}',
        source_filename TEXT,
        slide_count     INTEGER DEFAULT 0,
        total_criteria  INTEGER DEFAULT 0,
        created_at      TEXT NOT NULL,
        created_by      TEXT DEFAULT 'system',
        promoted_at     TEXT,
        changelog       TEXT NOT NULL DEFAULT '[]'
    )""",
    """CREATE TABLE IF NOT EXISTS deck_slides (
        id              INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        deck_id         TEXT NOT NULL REFERENCES deck_schemas(deck_id),
        slide_number    INTEGER NOT NULL,
        title           TEXT,
        raw_text        TEXT,
        speaker_notes   TEXT,
        slide_category  TEXT,
        section_id      TEXT,
        UNIQUE(deck_id, slide_number)
    )""",
    """CREATE TABLE IF NOT EXISTS deck_criteria (
        criterion_id        TEXT PRIMARY KEY,
        deck_id             TEXT NOT NULL REFERENCES deck_schemas(deck_id),
        slide_number        INTEGER NOT NULL,
        label               TEXT NOT NULL,
        explanation         TEXT NOT NULL,
        importance          TEXT NOT NULL DEFAULT 'should',
        slide_section       TEXT DEFAULT 'general',
        embedding_json      TEXT,
        detection_threshold DOUBLE PRECISION NOT NULL DEFAULT 0.72,
        disabled            INTEGER NOT NULL DEFAULT 0,
        human_override      TEXT,
        custom_notes        TEXT DEFAULT '',
        display_order       INTEGER DEFAULT 0
    )""",
    """CREATE TABLE IF NOT EXISTS deck_criteria_paraphrases (
        id              INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        criterion_id    TEXT NOT NULL REFERENCES deck_criteria(criterion_id),
        paraphrase_text TEXT NOT NULL,
        embedding_json  TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS deck_call_links (
        session_id  TEXT NOT NULL,
        deck_id     TEXT NOT NULL REFERENCES deck_schemas(deck_id),
        linked_at   TEXT NOT NULL,
        PRIMARY KEY (session_id, deck_id)
    )""",
    """CREATE TABLE IF NOT EXISTS deck_coverage_results (
        id              INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        session_id      TEXT NOT NULL,
        deck_id         TEXT NOT NULL,
        criterion_id    TEXT NOT NULL,
        covered         INTEGER NOT NULL DEFAULT 0,
        best_similarity DOUBLE PRECISION DEFAULT 0.0,
        best_window_idx INTEGER DEFAULT -1,
        detection_mode  TEXT DEFAULT 'direct',
        evaluated_at    TEXT NOT NULL,
        UNIQUE(session_id, deck_id, criterion_id)
    )""",
    """CREATE TABLE IF NOT EXISTS schema_edit_log (
        id           INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        deck_id      TEXT NOT NULL,
        criterion_id TEXT,
        field_name   TEXT NOT NULL,
        old_value    TEXT,
        new_value    TEXT,
        edited_by    TEXT DEFAULT 'admin',
        edited_at    TEXT NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS idx_criteria_deck ON deck_criteria(deck_id)",
    "CREATE INDEX IF NOT EXISTS idx_coverage_session ON deck_coverage_results(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_paraphrases_criterion ON deck_criteria_paraphrases(criterion_id)",
]

# schema_name -> ordered DDL statements (parent tables before child tables).
# Cross-schema FKs don't exist anywhere in this app, so only within-schema
# ordering matters — each module's own list above is already parent-first.
SCHEMA_DDL = [
    ("auth", AUTH),
    ("session_store", SESSION_STORE),
    ("activity_log", ACTIVITY_LOG),
    ("associate_roster", ASSOCIATE_ROSTER),
    ("job_queue", JOB_QUEUE),
    ("video_audits", VIDEO_AUDITS),
    ("deck", DECK),
]

_BY_NAME = dict(SCHEMA_DDL)


def apply_schema(conn, schema: str) -> None:
    """Create the given module's Postgres schema + all its tables/indexes
    if they don't already exist, then point the connection's search_path
    at it. `conn` only needs an `.execute(sql)` + `.commit()` method — works
    for both db.pg.PgConnCompat and a raw psycopg connection/cursor."""
    if schema not in _BY_NAME:
        raise ValueError(f"db/postgres_schema.py: unknown schema {schema!r}")
    conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
    conn.execute(f'SET search_path TO "{schema}", public')
    for stmt in _BY_NAME[schema]:
        conn.execute(stmt)
    conn.commit()
