"""
tests/test_auth_roles.py
Covers the two new granular roles (uploader / viewer) added alongside
admin/associate — uploader can only run new audits, viewer gets read-only
access to the full Audit History & Dashboard. See auth.py's VALID_ROLES
and api/main.py's require_role() for the server-side enforcement these
roles depend on (a role that can't see a page also can't call the API
behind it directly, not just a client-side redirect).
"""
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import auth


def _fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(auth, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(auth, "_local", threading.local())
    auth.init_db()


def test_all_four_roles_are_creatable(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch)
    for role in ("admin", "associate", "uploader", "viewer"):
        user = auth.create_user(email=f"{role}@kalvium.com", name=role.title(),
                                 password="pw12345678", role=role)
        assert user["role"] == role


def test_invalid_role_is_rejected(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        auth.create_user(email="x@kalvium.com", name="X", password="pw12345678", role="superadmin")


def test_seeded_admin_is_unaffected_by_new_roles(tmp_path, monkeypatch):
    # init_db() seeds the original shared admin login on first run — must
    # still work exactly as before regardless of the new roles existing.
    _fresh_db(tmp_path, monkeypatch)
    user = auth.authenticate("aryan@kalvium.com", "kalvium123")
    assert user is not None
    assert user["role"] == "admin"


def test_uploader_and_viewer_authenticate_like_any_other_role(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch)
    auth.create_user(email="up@kalvium.com", name="Up", password="pw12345678", role="uploader")
    auth.create_user(email="vw@kalvium.com", name="Vw", password="pw12345678", role="viewer")

    up = auth.authenticate("up@kalvium.com", "pw12345678")
    vw = auth.authenticate("vw@kalvium.com", "pw12345678")
    assert up["role"] == "uploader"
    assert vw["role"] == "viewer"


def test_disabled_uploader_cannot_authenticate(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch)
    user = auth.create_user(email="up@kalvium.com", name="Up", password="pw12345678", role="uploader")
    auth.update_user(user["user_id"], status="disabled")
    assert auth.authenticate("up@kalvium.com", "pw12345678") is None


# ── delete_user() ────────────────────────────────────────────────────────────

def test_delete_user_removes_the_account(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch)
    user = auth.create_user(email="gone@kalvium.com", name="Gone", password="pw12345678", role="viewer")
    assert auth.get_user(user["user_id"]) is not None

    auth.delete_user(user["user_id"])
    assert auth.get_user(user["user_id"]) is None
    assert auth.authenticate("gone@kalvium.com", "pw12345678") is None


def test_delete_user_also_kills_their_active_session(tmp_path, monkeypatch):
    # Deleting an account must sign it out everywhere immediately, not
    # leave a session token that still resolves until it happens to expire.
    _fresh_db(tmp_path, monkeypatch)
    user = auth.create_user(email="gone@kalvium.com", name="Gone", password="pw12345678", role="viewer")
    token = auth.create_session(user["user_id"])
    assert auth.get_session_user(token) is not None

    auth.delete_user(user["user_id"])
    assert auth.get_session_user(token) is None


def test_delete_user_does_not_affect_other_accounts(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch)
    keep = auth.create_user(email="keep@kalvium.com", name="Keep", password="pw12345678", role="admin")
    gone = auth.create_user(email="gone@kalvium.com", name="Gone", password="pw12345678", role="viewer")

    auth.delete_user(gone["user_id"])
    assert auth.get_user(keep["user_id"]) is not None
    assert auth.authenticate("keep@kalvium.com", "pw12345678") is not None
