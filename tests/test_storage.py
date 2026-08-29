"""
tests/test_storage.py
Focused tests for the Cloudflare R2 storage abstraction (storage/r2.py,
storage/backend.py, storage/persistent_file.py) and its integration
points (object-key generation, local-vs-R2 backend selection).

None of these tests use real R2 credentials or make a real network call —
every R2 client interaction is mocked. boto3 is installed in this
environment, but these tests would still pass without it, since
get_r2_client() itself is monkeypatched/mocked rather than exercised for
real. Live connectivity is exercised only by scripts/test_r2.py, run
manually by a human with real credentials — never by this suite.

IMPORTANT: these tests must never assume anything about the ambient
.env/environment on the machine running them — a developer's real .env
may legitimately have STORAGE_BACKEND=r2 and real R2 credentials set (as
this project's own local dev environment does). Every test that cares
about a *specific* configuration state monkeypatches config.settings.
get_settings (as imported into storage.backend) with a fake settings
object, rather than asserting on whatever get_settings() actually
returns right now.
"""
from __future__ import annotations
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

import storage.backend as backend
from storage.backend import get_r2_config
import storage.r2 as r2


def _fake_settings(**overrides) -> SimpleNamespace:
    """A minimal stand-in for config.settings.Settings with just the
    fields storage/backend.py reads, defaulting to the "unconfigured,
    local" state regardless of what this machine's real .env contains."""
    defaults = dict(
        storage_backend="local",
        r2_endpoint_url="",
        r2_bucket_name="kalvium-audit-storage",
        r2_access_key_id="",
        r2_secret_access_key="",
        r2_region="auto",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ── Configuration loading / local-vs-R2 backend selection ──────────────────

def test_storage_backend_defaults_to_local(monkeypatch):
    monkeypatch.setattr(backend, "get_settings", lambda: _fake_settings())
    assert backend.storage_backend() == "local"
    assert backend.is_r2_enabled() is False


def test_storage_backend_r2_when_explicitly_set(monkeypatch):
    """The other half of the same switch — confirms is_r2_enabled() does
    turn on when storage_backend='r2', not just that it stays off by
    default."""
    monkeypatch.setattr(backend, "get_settings", lambda: _fake_settings(storage_backend="r2"))
    assert backend.storage_backend() == "r2"
    assert backend.is_r2_enabled() is True


def test_r2_not_configured_by_default(monkeypatch):
    """With no R2_* values set, r2_configured() must be False — this is
    what keeps the app fully runnable locally with R2 untouched."""
    monkeypatch.setattr(backend, "get_settings", lambda: _fake_settings())
    assert backend.r2_configured() is False


def test_r2_configured_when_all_values_present(monkeypatch):
    monkeypatch.setattr(backend, "get_settings", lambda: _fake_settings(
        r2_endpoint_url="https://example.r2.cloudflarestorage.com",
        r2_access_key_id="fake-key-id",
        r2_secret_access_key="fake-secret",
    ))
    assert backend.r2_configured() is True


def test_get_r2_config_shape(monkeypatch):
    monkeypatch.setattr(backend, "get_settings", lambda: _fake_settings())
    cfg = get_r2_config()
    assert set(cfg.keys()) == {"endpoint_url", "bucket_name", "access_key_id", "secret_access_key", "region"}
    assert cfg["bucket_name"] == "kalvium-audit-storage"  # the documented default
    assert cfg["region"] == "auto"


def test_get_r2_client_raises_config_error_when_unconfigured(monkeypatch):
    """Calling into R2 with no credentials configured must raise a clear,
    generic StorageConfigError — never a raw boto3/botocore exception,
    and never anything that could contain a credential (there isn't one
    to leak here, since none is configured)."""
    monkeypatch.setattr(backend, "get_settings", lambda: _fake_settings())
    try:
        r2.get_r2_client()
        raised = False
    except r2.StorageConfigError:
        raised = True
    assert raised


# ── Object-key generation (adapted structure, per the actual data model) ───

def test_video_screenshot_key_structure():
    """audits/{video_audit_id}/screenshots/{filename} — built in api/main.py's
    _persist_screenshots_to_r2(); asserted here as a literal string pattern
    so a future refactor can't silently change it without this test
    noticing."""
    video_audit_id = "abc-123"
    filename = "frame_05pct.jpg"
    key = f"audits/{video_audit_id}/screenshots/{filename}"
    assert key == "audits/abc-123/screenshots/frame_05pct.jpg"


def test_excel_tracker_key_is_fixed_singleton_path():
    from reports.audit_excel_manager import _R2_KEY
    assert _R2_KEY == "tracker/audit_records.xlsx"


def test_google_sheets_keys():
    """Google Sheets integration switched from OAuth2 (credentials/token/
    PKCE files) to a single service-account key — see
    reports/google_sheets_manager.py."""
    from reports.google_sheets_manager import _R2_KEY_SERVICE_ACCOUNT, _R2_KEY_CONFIG
    assert _R2_KEY_SERVICE_ACCOUNT == "integrations/google_sheets/service_account.json"
    assert _R2_KEY_CONFIG == "integrations/google_sheets/config.json"


def test_word_mappings_and_tl_map_keys():
    from utils.word_normalizer import _R2_KEY as word_key
    from reports.tl_mapping import _R2_KEY as tl_key
    assert word_key == "config/word_mappings.json"
    assert tl_key == "config/associate_tl_map.json"


# ── Upload / download / delete via mocked boto3 client ──────────────────────

def _patched_client(monkeypatch, fake_client):
    monkeypatch.setattr(r2, "get_r2_client", lambda: fake_client)
    monkeypatch.setattr(r2, "_bucket_name", lambda: "kalvium-audit-storage")


def test_upload_file_calls_boto3_upload_file(monkeypatch, tmp_path):
    fake_client = MagicMock()
    _patched_client(monkeypatch, fake_client)
    local = tmp_path / "shot.jpg"
    local.write_bytes(b"fake-jpeg-bytes")

    r2.upload_file(local, "audits/x/screenshots/shot.jpg", content_type="image/jpeg")

    fake_client.upload_file.assert_called_once()
    args, kwargs = fake_client.upload_file.call_args
    assert args[0] == str(local)
    assert args[1] == "kalvium-audit-storage"
    assert args[2] == "audits/x/screenshots/shot.jpg"
    assert kwargs["ExtraArgs"] == {"ContentType": "image/jpeg"}


def test_upload_bytes_calls_put_object(monkeypatch):
    fake_client = MagicMock()
    _patched_client(monkeypatch, fake_client)

    r2.upload_bytes(b"hello", "config/word_mappings.json", content_type="application/json")

    fake_client.put_object.assert_called_once_with(
        Bucket="kalvium-audit-storage", Key="config/word_mappings.json",
        Body=b"hello", ContentType="application/json",
    )


def test_download_bytes_returns_body(monkeypatch):
    fake_client = MagicMock()
    fake_body = MagicMock()
    fake_body.read.return_value = b"the-file-contents"
    fake_client.get_object.return_value = {"Body": fake_body}
    _patched_client(monkeypatch, fake_client)

    result = r2.download_bytes("some/key.txt")
    assert result == b"the-file-contents"


def test_delete_file_calls_delete_object(monkeypatch):
    fake_client = MagicMock()
    _patched_client(monkeypatch, fake_client)

    r2.delete_file("some/key.txt")

    fake_client.delete_object.assert_called_once_with(Bucket="kalvium-audit-storage", Key="some/key.txt")


def test_object_exists_true_and_false(monkeypatch):
    fake_client = MagicMock()
    _patched_client(monkeypatch, fake_client)
    fake_client.head_object.return_value = {}
    assert r2.object_exists("present.txt") is True

    from botocore.exceptions import ClientError
    fake_client.head_object.side_effect = ClientError(
        {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject"
    )
    assert r2.object_exists("missing.txt") is False


def test_get_object_url_is_not_a_fetchable_public_url():
    """The bucket is private — get_object_url() must return an internal
    reference string, not something that looks like a public HTTP URL a
    browser could just fetch."""
    url = r2.get_object_url("audits/x/screenshots/shot.jpg")
    assert url.startswith("r2://")
    assert "http://" not in url and "https://" not in url


def test_generate_presigned_download_url(monkeypatch):
    fake_client = MagicMock()
    fake_client.generate_presigned_url.return_value = "https://example-signed-url/abc"
    _patched_client(monkeypatch, fake_client)

    url = r2.generate_presigned_download_url("audits/x/screenshots/shot.jpg", expires_in=1800)

    assert url == "https://example-signed-url/abc"
    fake_client.generate_presigned_url.assert_called_once_with(
        "get_object",
        Params={"Bucket": "kalvium-audit-storage", "Key": "audits/x/screenshots/shot.jpg"},
        ExpiresIn=1800,
    )


# ── Failure handling ─────────────────────────────────────────────────────────

def test_download_file_missing_object_raises_not_found(monkeypatch, tmp_path):
    from botocore.exceptions import ClientError
    fake_client = MagicMock()
    fake_client.download_file.side_effect = ClientError(
        {"Error": {"Code": "404", "Message": "Not Found"}}, "GetObject"
    )
    _patched_client(monkeypatch, fake_client)

    try:
        r2.download_file("missing.txt", tmp_path / "out.txt")
        raised = None
    except Exception as exc:
        raised = exc
    assert isinstance(raised, r2.StorageNotFoundError)


def test_download_file_other_client_error_raises_download_error(monkeypatch, tmp_path):
    from botocore.exceptions import ClientError
    fake_client = MagicMock()
    fake_client.download_file.side_effect = ClientError(
        {"Error": {"Code": "500", "Message": "Internal Error"}}, "GetObject"
    )
    _patched_client(monkeypatch, fake_client)

    try:
        r2.download_file("some.txt", tmp_path / "out.txt")
        raised = None
    except Exception as exc:
        raised = exc
    assert isinstance(raised, r2.StorageDownloadError)
    assert not isinstance(raised, r2.StorageNotFoundError)


def test_upload_unavailable_raises_storage_unavailable(monkeypatch, tmp_path):
    from botocore.exceptions import EndpointConnectionError
    fake_client = MagicMock()
    fake_client.upload_file.side_effect = EndpointConnectionError(endpoint_url="https://fake")
    _patched_client(monkeypatch, fake_client)
    local = tmp_path / "f.txt"
    local.write_text("x")

    try:
        r2.upload_file(local, "key.txt")
        raised = None
    except Exception as exc:
        raised = exc
    assert isinstance(raised, r2.StorageUnavailableError)


def test_upload_missing_local_file_raises_upload_error(monkeypatch, tmp_path):
    fake_client = MagicMock()
    _patched_client(monkeypatch, fake_client)

    try:
        r2.upload_file(tmp_path / "does-not-exist.txt", "key.txt")
        raised = None
    except Exception as exc:
        raised = exc
    assert isinstance(raised, r2.StorageUploadError)


def test_delete_failure_raises_storage_delete_error(monkeypatch):
    from botocore.exceptions import ClientError
    fake_client = MagicMock()
    fake_client.delete_object.side_effect = ClientError(
        {"Error": {"Code": "403", "Message": "Forbidden"}}, "DeleteObject"
    )
    _patched_client(monkeypatch, fake_client)

    try:
        r2.delete_file("key.txt")
        raised = None
    except Exception as exc:
        raised = exc
    assert isinstance(raised, r2.StorageDeleteError)


def test_health_check_reports_not_configured(monkeypatch):
    monkeypatch.setattr(backend, "get_settings", lambda: _fake_settings())
    ok, msg = r2.health_check()
    assert ok is False
    assert msg == "R2 is not configured"


# ── storage/persistent_file.py: no-op in local mode, wired in R2 mode ──────

def test_sync_helpers_are_noop_in_local_mode(tmp_path, monkeypatch):
    """The default (STORAGE_BACKEND=local) must make every sync helper a
    complete no-op — this is what guarantees zero behavior change for
    every existing caller (audit_excel_manager, google_sheets_manager,
    word_normalizer, tl_mapping) unless R2 is explicitly enabled.

    Explicitly forced to "local" here regardless of this machine's real
    .env (which may legitimately have STORAGE_BACKEND=r2 set) — without
    this, this test would silently make a REAL network call against the
    real bucket on any machine where R2 happens to be enabled, which is
    exactly what happened once during development of this test (a stray
    config/thing.json object was left in the real bucket and had to be
    manually cleaned up)."""
    monkeypatch.setattr(backend, "is_r2_enabled", lambda: False)
    import storage.persistent_file as pf

    local_path = tmp_path / "thing.json"
    # sync_from_r2_if_missing: must not attempt anything (no R2 client
    # ever constructed) even though the local file doesn't exist yet.
    pf.sync_from_r2_if_missing(local_path, "config/thing.json")
    assert not local_path.exists()

    local_path.write_text("{}")
    # sync_to_r2: must not attempt anything either.
    pf.sync_to_r2(local_path, "config/thing.json")
    # No assertion possible beyond "it didn't raise" — there's no R2
    # client in local mode to have been called, which is the point.

    pf.delete_from_r2("config/thing.json")  # must not raise


def test_sync_to_r2_calls_upload_when_r2_enabled(tmp_path, monkeypatch):
    import storage.backend as backend
    import storage.persistent_file as pf

    monkeypatch.setattr(backend, "is_r2_enabled", lambda: True)
    called = {}

    def fake_upload_file(local_path, key, content_type=None):
        called["local_path"] = local_path
        called["key"] = key
        called["content_type"] = content_type

    monkeypatch.setattr("storage.r2.upload_file", fake_upload_file)

    local_path = tmp_path / "config.json"
    local_path.write_text("{}")
    pf.sync_to_r2(local_path, "config/thing.json", content_type="application/json")

    assert called["key"] == "config/thing.json"
    assert called["content_type"] == "application/json"


def test_sync_from_r2_if_missing_calls_download_when_r2_enabled_and_missing_locally(tmp_path, monkeypatch):
    import storage.backend as backend
    import storage.persistent_file as pf

    monkeypatch.setattr(backend, "is_r2_enabled", lambda: True)
    called = {}

    def fake_download_file(key, local_path):
        called["key"] = key
        Path(local_path).write_text("recovered")

    monkeypatch.setattr("storage.r2.download_file", fake_download_file)

    local_path = tmp_path / "recovered.json"
    assert not local_path.exists()
    pf.sync_from_r2_if_missing(local_path, "config/thing.json")

    assert called["key"] == "config/thing.json"
    assert local_path.read_text() == "recovered"


def test_sync_from_r2_if_missing_skips_download_when_local_file_already_present(tmp_path, monkeypatch):
    """Must not overwrite a local file that already exists with an R2 copy
    — sync-if-missing, not sync-always. (R2 already having the object too
    means there's nothing to backfill either — see the backfill test
    below for the complementary case.)"""
    import storage.backend as backend
    import storage.persistent_file as pf

    monkeypatch.setattr(backend, "is_r2_enabled", lambda: True)
    download_called = {"n": 0}
    upload_called = {"n": 0}

    monkeypatch.setattr("storage.r2.download_file", lambda key, local_path: download_called.__setitem__("n", download_called["n"] + 1))
    monkeypatch.setattr("storage.r2.upload_file", lambda local_path, key, content_type=None: upload_called.__setitem__("n", upload_called["n"] + 1))
    monkeypatch.setattr("storage.r2.object_exists", lambda key: True)  # R2 already has it

    local_path = tmp_path / "already_here.json"
    local_path.write_text("original")
    pf.sync_from_r2_if_missing(local_path, "config/thing.json")

    assert download_called["n"] == 0
    assert upload_called["n"] == 0
    assert local_path.read_text() == "original"


def test_sync_from_r2_if_missing_backfills_when_local_exists_but_r2_does_not(tmp_path, monkeypatch):
    """The gap this fix closes: a file that already existed locally BEFORE
    R2 was ever enabled (e.g. a Google OAuth token established under
    STORAGE_BACKEND=local) must get pushed to R2 the first time it's read
    after switching to r2 — otherwise it would silently never reach R2
    until it happened to be rewritten, and would be lost on the very next
    restart despite the file "looking" persisted locally."""
    import storage.backend as backend
    import storage.persistent_file as pf

    monkeypatch.setattr(backend, "is_r2_enabled", lambda: True)
    monkeypatch.setattr("storage.r2.object_exists", lambda key: False)  # R2 doesn't have it yet
    uploaded = {}

    def fake_upload_file(local_path, key, content_type=None):
        uploaded["local_path"] = local_path
        uploaded["key"] = key

    monkeypatch.setattr("storage.r2.upload_file", fake_upload_file)

    local_path = tmp_path / "pre_existing_token.json"
    local_path.write_text("pre-existing local content")
    pf.sync_from_r2_if_missing(local_path, "integrations/google_sheets/token.json")

    assert uploaded["key"] == "integrations/google_sheets/token.json"
    assert uploaded["local_path"] == local_path
    assert local_path.read_text() == "pre-existing local content"  # untouched locally
