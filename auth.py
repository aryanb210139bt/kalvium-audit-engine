"""
auth.py
Real per-person accounts, replacing the old shared hardcoded login.

- Passwords: PBKDF2-HMAC-SHA256 (stdlib hashlib + secrets — no new
  dependency), 200k iterations, random salt per user.
- Sessions: opaque random token in an HttpOnly cookie, looked up against a
  server-side table. Matches the "one server, small team" hosting model —
  no need for stateless JWTs across multiple server instances.
- Roles: "admin" (sees everything, manages users/decks/tracker) and
  "associate" (sees only their own audits — matched via associate_name
  against the same "Lead Owner" field used elsewhere in the app).
- Accounts are admin-granted only — there is no self-signup endpoint.

On first run, seeds one admin account matching the previous shared login
(aryan@kalvium.com) so existing access isn't lost. Change that password
via the Manage Users page once you're in.
"""
from __future__ import annotations
import hashlib
import secrets
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent / "data" / "auth.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

SESSION_COOKIE   = "kda_session"
SESSION_TTL_DAYS = 14
PBKDF2_ITERS     = 200_000

_local = threading.local()


def _conn():
    from db.backend import is_postgres_enabled, get_database_url
    if is_postgres_enabled():
        from db.pg import get_pg_connection
        # "kalvium_auth", NOT "auth" — a managed Postgres provider (e.g.
        # Supabase) commonly pre-provisions its OWN "auth" schema for its
        # own built-in user system (confirmed live: Supabase's real "auth"
        # schema already has 22 tables, including ones named `users` and
        # `sessions`, structurally unrelated to ours). Using the bare word
        # "auth" here would target/collide with that instead of our own
        # tables. Never rename this back to "auth".
        return get_pg_connection(get_database_url(), "kalvium_auth")
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
    return _local.conn


def init_db() -> None:
    from db.backend import is_postgres_enabled
    if is_postgres_enabled():
        from db.postgres_schema import apply_schema
        apply_schema(_conn(), "kalvium_auth")
        _seed_initial_admin()
        return
    db = _conn()
    db.executescript("""
    CREATE TABLE IF NOT EXISTS users (
        user_id         TEXT PRIMARY KEY,
        email           TEXT UNIQUE NOT NULL,
        name            TEXT NOT NULL,
        password_hash   TEXT NOT NULL,
        password_salt   TEXT NOT NULL,
        role            TEXT NOT NULL DEFAULT 'associate',   -- admin | associate
        associate_name  TEXT DEFAULT '',                      -- links to session "Lead Owner" for scoping
        status          TEXT NOT NULL DEFAULT 'active',       -- active | disabled
        created_at      TEXT NOT NULL,
        last_login_at   TEXT
    );
    CREATE TABLE IF NOT EXISTS sessions (
        token       TEXT PRIMARY KEY,
        user_id     TEXT NOT NULL REFERENCES users(user_id),
        created_at  TEXT NOT NULL,
        expires_at  TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
    """)
    db.commit()
    _seed_initial_admin()


def _seed_initial_admin() -> None:
    """First-run only — preserves access for the account CLAUDE.md documents
    as the shared login. Safe to call every startup (no-ops if any user exists)."""
    db = _conn()
    if db.execute("SELECT 1 FROM users LIMIT 1").fetchone():
        return
    create_user(email="aryan@kalvium.com", name="Aryan", password="kalvium123", role="admin")


# ── Password hashing ─────────────────────────────────────────────────────────

def _hash_password(password: str, salt: Optional[str] = None) -> tuple[str, str]:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), PBKDF2_ITERS).hex()
    return digest, salt


def _verify_password(password: str, digest: str, salt: str) -> bool:
    check, _ = _hash_password(password, salt)
    return secrets.compare_digest(check, digest)


# ── Users ─────────────────────────────────────────────────────────────────────

def create_user(email: str, name: str, password: str, role: str = "associate",
                 associate_name: str = "") -> dict:
    if role not in ("admin", "associate"):
        raise ValueError("role must be 'admin' or 'associate'")
    email = email.strip().lower()
    digest, salt = _hash_password(password)
    user_id = str(uuid.uuid4())
    db = _conn()
    db.execute(
        "INSERT INTO users (user_id, email, name, password_hash, password_salt, role, "
        "associate_name, status, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (user_id, email, name, digest, salt, role, associate_name, "active",
         datetime.utcnow().isoformat()),
    )
    db.commit()
    return get_user(user_id)


def list_users() -> list[dict]:
    rows = _conn().execute(
        "SELECT user_id, email, name, role, associate_name, status, created_at, last_login_at "
        "FROM users ORDER BY created_at"
    ).fetchall()
    return [dict(r) for r in rows]


def get_user(user_id: str) -> Optional[dict]:
    row = _conn().execute(
        "SELECT user_id, email, name, role, associate_name, status, created_at, last_login_at "
        "FROM users WHERE user_id=?", (user_id,)
    ).fetchone()
    return dict(row) if row else None


def update_user(user_id: str, **fields) -> None:
    """fields: any of role, associate_name, status, name — password handled separately."""
    allowed = {"role", "associate_name", "status", "name"}
    updates = {k: v for k, v in fields.items() if k in allowed and v is not None}
    if not updates:
        return
    set_clause = ", ".join(f"{k}=?" for k in updates)
    db = _conn()
    db.execute(f"UPDATE users SET {set_clause} WHERE user_id=?", (*updates.values(), user_id))
    db.commit()


def reset_password(user_id: str, new_password: str) -> None:
    digest, salt = _hash_password(new_password)
    db = _conn()
    db.execute("UPDATE users SET password_hash=?, password_salt=? WHERE user_id=?",
               (digest, salt, user_id))
    db.commit()


# ── Login / sessions ──────────────────────────────────────────────────────────

def authenticate(email: str, password: str) -> Optional[dict]:
    """Verify credentials. Returns the user dict (without password fields) or None."""
    email = email.strip().lower()
    row = _conn().execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    if not row:
        return None
    row = dict(row)
    if row["status"] != "active":
        return None
    if not _verify_password(password, row["password_hash"], row["password_salt"]):
        return None
    db = _conn()
    db.execute("UPDATE users SET last_login_at=? WHERE user_id=?",
               (datetime.utcnow().isoformat(), row["user_id"]))
    db.commit()
    return get_user(row["user_id"])


def create_session(user_id: str) -> str:
    token = secrets.token_urlsafe(32)
    now = datetime.utcnow()
    db = _conn()
    db.execute("INSERT INTO sessions (token, user_id, created_at, expires_at) VALUES (?,?,?,?)",
               (token, user_id, now.isoformat(), (now + timedelta(days=SESSION_TTL_DAYS)).isoformat()))
    db.commit()
    return token


def get_session_user(token: str) -> Optional[dict]:
    if not token:
        return None
    row = _conn().execute("SELECT * FROM sessions WHERE token=?", (token,)).fetchone()
    if not row:
        return None
    if row["expires_at"] < datetime.utcnow().isoformat():
        delete_session(token)
        return None
    user = get_user(row["user_id"])
    if not user or user["status"] != "active":
        return None
    return user


def delete_session(token: str) -> None:
    db = _conn()
    db.execute("DELETE FROM sessions WHERE token=?", (token,))
    db.commit()
