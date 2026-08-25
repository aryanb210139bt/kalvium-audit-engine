"""
deck/db.py
SQLite database for the dynamic deck evaluation system.
All deck schemas, criteria, embeddings, and coverage results live here.
"""
from __future__ import annotations
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).parent.parent / "data" / "decks.db"
DB_PATH.parent.mkdir(exist_ok=True)

_local = threading.local()


def _conn() -> sqlite3.Connection:
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA foreign_keys=ON")
    return _local.conn


def init_db():
    """Create all tables if they don't exist."""
    db = _conn()
    db.executescript("""
    -- ── Deck schemas ──────────────────────────────────────────────────────────
    CREATE TABLE IF NOT EXISTS deck_schemas (
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
    );

    -- ── Slides ─────────────────────────────────────────────────────────────────
    CREATE TABLE IF NOT EXISTS deck_slides (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        deck_id         TEXT NOT NULL REFERENCES deck_schemas(deck_id),
        slide_number    INTEGER NOT NULL,
        title           TEXT,
        raw_text        TEXT,
        speaker_notes   TEXT,
        slide_category  TEXT,
        section_id      TEXT,
        UNIQUE(deck_id, slide_number)
    );

    -- ── Criteria ───────────────────────────────────────────────────────────────
    CREATE TABLE IF NOT EXISTS deck_criteria (
        criterion_id        TEXT PRIMARY KEY,
        deck_id             TEXT NOT NULL REFERENCES deck_schemas(deck_id),
        slide_number        INTEGER NOT NULL,
        label               TEXT NOT NULL,
        explanation         TEXT NOT NULL,
        importance          TEXT NOT NULL DEFAULT 'should',
        slide_section       TEXT DEFAULT 'general',
        embedding_json      TEXT,
        detection_threshold REAL NOT NULL DEFAULT 0.72,
        disabled            INTEGER NOT NULL DEFAULT 0,
        human_override      TEXT,
        custom_notes        TEXT DEFAULT '',
        display_order       INTEGER DEFAULT 0
    );

    -- ── Paraphrase embeddings ──────────────────────────────────────────────────
    CREATE TABLE IF NOT EXISTS deck_criteria_paraphrases (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        criterion_id    TEXT NOT NULL REFERENCES deck_criteria(criterion_id),
        paraphrase_text TEXT NOT NULL,
        embedding_json  TEXT
    );

    -- ── Call ↔ Deck linkage (frozen at processing time) ────────────────────────
    CREATE TABLE IF NOT EXISTS deck_call_links (
        session_id  TEXT NOT NULL,
        deck_id     TEXT NOT NULL REFERENCES deck_schemas(deck_id),
        linked_at   TEXT NOT NULL,
        PRIMARY KEY (session_id, deck_id)
    );

    -- ── Per-session coverage results ───────────────────────────────────────────
    CREATE TABLE IF NOT EXISTS deck_coverage_results (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id      TEXT NOT NULL,
        deck_id         TEXT NOT NULL,
        criterion_id    TEXT NOT NULL,
        covered         INTEGER NOT NULL DEFAULT 0,
        best_similarity REAL DEFAULT 0.0,
        best_window_idx INTEGER DEFAULT -1,
        detection_mode  TEXT DEFAULT 'direct',
        evaluated_at    TEXT NOT NULL,
        UNIQUE(session_id, deck_id, criterion_id)
    );

    -- ── Human edit audit log ───────────────────────────────────────────────────
    CREATE TABLE IF NOT EXISTS schema_edit_log (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        deck_id     TEXT NOT NULL,
        criterion_id TEXT,
        field_name  TEXT NOT NULL,
        old_value   TEXT,
        new_value   TEXT,
        edited_by   TEXT DEFAULT 'admin',
        edited_at   TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_criteria_deck ON deck_criteria(deck_id);
    CREATE INDEX IF NOT EXISTS idx_coverage_session ON deck_coverage_results(session_id);
    CREATE INDEX IF NOT EXISTS idx_paraphrases_criterion ON deck_criteria_paraphrases(criterion_id);
    """)
    db.commit()


# ── CRUD helpers ───────────────────────────────────────────────────────────────

def save_deck_schema(schema: dict):
    db = _conn()
    db.execute("""
        INSERT OR REPLACE INTO deck_schemas
        (deck_id, deck_slug, version_major, schema_patch, deck_name, status,
         scoring_weights, source_filename, slide_count, total_criteria,
         created_at, created_by, changelog)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        schema["deck_id"], schema["deck_slug"],
        schema.get("version_major", 1), schema.get("schema_patch", 0),
        schema["deck_name"], schema.get("status", "draft"),
        json.dumps(schema.get("scoring_weights", {"behavioral_layer": 0.55, "deck_coverage_layer": 0.45})),
        schema.get("source_filename"), schema.get("slide_count", 0),
        schema.get("total_criteria", 0), schema["created_at"],
        schema.get("created_by", "system"),
        json.dumps(schema.get("changelog", [])),
    ))
    db.commit()


def save_slide(slide: dict):
    db = _conn()
    db.execute("""
        INSERT OR REPLACE INTO deck_slides
        (deck_id, slide_number, title, raw_text, speaker_notes, slide_category, section_id)
        VALUES (?,?,?,?,?,?,?)
    """, (
        slide["deck_id"], slide["slide_number"], slide.get("title", ""),
        slide.get("raw_text", ""), slide.get("speaker_notes", ""),
        slide.get("slide_category", "misc"), slide.get("section_id", "general"),
    ))
    db.commit()


def save_criterion(c: dict):
    db = _conn()
    db.execute("""
        INSERT OR REPLACE INTO deck_criteria
        (criterion_id, deck_id, slide_number, label, explanation, importance,
         slide_section, embedding_json, detection_threshold, disabled,
         human_override, custom_notes, display_order)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        c["criterion_id"], c["deck_id"], c["slide_number"],
        c["label"], c["explanation"], c.get("importance", "should"),
        c.get("slide_section", "general"),
        json.dumps(c["embedding"]) if c.get("embedding") else None,
        c.get("detection_threshold", 0.72), int(c.get("disabled", False)),
        c.get("human_override"), c.get("custom_notes", ""),
        c.get("display_order", 0),
    ))
    db.commit()


def save_paraphrase(criterion_id: str, text: str, embedding: list | None):
    db = _conn()
    db.execute("""
        INSERT INTO deck_criteria_paraphrases (criterion_id, paraphrase_text, embedding_json)
        VALUES (?,?,?)
    """, (criterion_id, text, json.dumps(embedding) if embedding else None))
    db.commit()


def get_all_decks() -> list[dict]:
    db = _conn()
    rows = db.execute("""
        SELECT ds.deck_id, ds.deck_slug, ds.version_major, ds.schema_patch, ds.deck_name,
               ds.status, ds.source_filename, ds.slide_count, ds.created_at,
               COUNT(dc.criterion_id) AS total_criteria
        FROM deck_schemas ds
        LEFT JOIN deck_criteria dc ON dc.deck_id = ds.deck_id
        GROUP BY ds.deck_id
        ORDER BY ds.created_at DESC
    """).fetchall()
    return [dict(r) for r in rows]


def get_deck(deck_id: str) -> dict | None:
    db = _conn()
    row = db.execute("""
        SELECT ds.*, COUNT(dc.criterion_id) AS total_criteria
        FROM deck_schemas ds
        LEFT JOIN deck_criteria dc ON dc.deck_id = ds.deck_id
        WHERE ds.deck_id = ?
        GROUP BY ds.deck_id
    """, (deck_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["scoring_weights"] = json.loads(d["scoring_weights"])
    d["changelog"] = json.loads(d["changelog"])
    return d


def get_deck_slides(deck_id: str) -> list[dict]:
    db = _conn()
    return [dict(r) for r in db.execute(
        "SELECT * FROM deck_slides WHERE deck_id=? ORDER BY slide_number", (deck_id,)
    ).fetchall()]


def get_deck_criteria(deck_id: str, include_disabled: bool = False) -> list[dict]:
    db = _conn()
    q = "SELECT * FROM deck_criteria WHERE deck_id=?"
    if not include_disabled:
        q += " AND disabled=0"
    q += " ORDER BY slide_number, display_order"
    rows = db.execute(q, (deck_id,)).fetchall()
    result = []
    for r in rows:
        c = dict(r)
        c["embedding"] = json.loads(c["embedding_json"]) if c.get("embedding_json") else None
        del c["embedding_json"]
        c["disabled"] = bool(c["disabled"])
        result.append(c)
    return result


def get_criterion_paraphrases(criterion_id: str) -> list[dict]:
    db = _conn()
    rows = db.execute(
        "SELECT * FROM deck_criteria_paraphrases WHERE criterion_id=?", (criterion_id,)
    ).fetchall()
    result = []
    for r in rows:
        p = dict(r)
        p["embedding"] = json.loads(p["embedding_json"]) if p.get("embedding_json") else None
        result.append(p)
    return result


def get_active_deck() -> dict | None:
    db = _conn()
    row = db.execute(
        "SELECT * FROM deck_schemas WHERE status='active' ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    if not row:
        return None
    d = dict(row)
    d["scoring_weights"] = json.loads(d["scoring_weights"])
    return d


def update_deck_status(deck_id: str, status: str, promoted_at: str = None):
    db = _conn()
    if promoted_at:
        db.execute(
            "UPDATE deck_schemas SET status=?, promoted_at=? WHERE deck_id=?",
            (status, promoted_at, deck_id)
        )
    else:
        db.execute("UPDATE deck_schemas SET status=? WHERE deck_id=?", (status, deck_id))
    db.commit()


def update_criterion(criterion_id: str, fields: dict, edited_by: str = "admin"):
    db = _conn()
    from datetime import datetime
    for field, new_val in fields.items():
        old_row = db.execute(
            f"SELECT {field} FROM deck_criteria WHERE criterion_id=?", (criterion_id,)
        ).fetchone()
        old_val = dict(old_row)[field] if old_row else None
        db.execute(
            "INSERT INTO schema_edit_log (deck_id, criterion_id, field_name, old_value, new_value, edited_by, edited_at) "
            "SELECT deck_id, ?, ?, ?, ?, ?, ? FROM deck_criteria WHERE criterion_id=?",
            (criterion_id, field, str(old_val), str(new_val), edited_by,
             datetime.utcnow().isoformat(), criterion_id)
        )
        db.execute(
            f"UPDATE deck_criteria SET {field}=? WHERE criterion_id=?", (new_val, criterion_id)
        )
    db.commit()


def save_coverage_results(session_id: str, deck_id: str, results: list[dict]):
    db = _conn()
    from datetime import datetime
    now = datetime.utcnow().isoformat()
    for r in results:
        db.execute("""
            INSERT OR REPLACE INTO deck_coverage_results
            (session_id, deck_id, criterion_id, covered, best_similarity,
             best_window_idx, detection_mode, evaluated_at)
            VALUES (?,?,?,?,?,?,?,?)
        """, (
            session_id, deck_id, r["criterion_id"],
            int(r.get("covered", False)), r.get("best_similarity", 0.0),
            r.get("best_window_idx", -1), r.get("detection_mode", "direct"), now,
        ))
    db.commit()


def get_coverage_results(session_id: str, deck_id: str) -> list[dict]:
    db = _conn()
    return [dict(r) for r in db.execute(
        "SELECT * FROM deck_coverage_results WHERE session_id=? AND deck_id=?",
        (session_id, deck_id)
    ).fetchall()]


def link_call_to_deck(session_id: str, deck_id: str):
    db = _conn()
    from datetime import datetime
    db.execute(
        "INSERT OR IGNORE INTO deck_call_links (session_id, deck_id, linked_at) VALUES (?,?,?)",
        (session_id, deck_id, datetime.utcnow().isoformat())
    )
    db.commit()


# Initialise on import
init_db()
