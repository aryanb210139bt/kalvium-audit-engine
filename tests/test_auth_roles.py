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
