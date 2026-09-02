"""
tests/test_sarvam_job_store.py
Unit tests for transcription/sarvam_job_store.py — the durable per-session
record of Sarvam Batch STT jobs. conftest.py's autouse fixture already
forces DB_BACKEND=sqlite/STORAGE_BACKEND=local for every test in this
suite; each test here additionally points DB_PATH at a fresh temp file so
tests never share state with each other or with the real data/ directory.
"""
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import transcription.sarvam_job_store as store


def _fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "sarvam_stt_jobs.db")
    monkeypatch.setattr(store, "_local", threading.local())
    store.init_db()


def test_upsert_submitted_creates_row(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch)
    store.upsert_submitted("s1", "job-abc", "/tmp/a.wav", 5400.0, 1, now="2026-01-01T00:00:00")
    row = store.get("s1")
    assert row["sarvam_job_id"] == "job-abc"
    assert row["status"] == "submitted"
    assert row["wav_path"] == "/tmp/a.wav"
    assert row["duration_seconds"] == 5400.0
    assert row["retry_count"] == 0
    assert row["cancel_requested"] == 0


def test_upsert_submitted_twice_increments_retry_and_resets_cancel(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch)
    store.upsert_submitted("s1", "job-1", "/tmp/a.wav", 100.0, 1, now="t0")
    store.request_cancel("s1", now="t1")
    assert store.is_cancel_requested("s1") is True

    # A fresh submission (e.g. a full resubmit after a permanent failure)
    # increments retry_count and clears any stale cancel flag.
    store.upsert_submitted("s1", "job-2", "/tmp/a.wav", 100.0, 1, now="t2")
    row = store.get("s1")
    assert row["sarvam_job_id"] == "job-2"
    assert row["status"] == "submitted"
    assert row["retry_count"] == 1
    assert row["cancel_requested"] == 0


def test_update_status_sets_error_message(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch)
    store.upsert_submitted("s1", "job-1", "/tmp/a.wav", 100.0, 1, now="t0")
    store.update_status("s1", "stt_failed", "boom", now="t1")
    row = store.get("s1")
    assert row["status"] == "stt_failed"
    assert row["error_message"] == "boom"


def test_get_missing_session_returns_none(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch)
    assert store.get("does-not-exist") is None


def test_request_cancel_blocked_on_terminal_status(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch)
    store.upsert_submitted("s1", "job-1", "/tmp/a.wav", 100.0, 1, now="t0")
    store.update_status("s1", "completed", now="t1")
    ok = store.request_cancel("s1", now="t2")
    assert ok is False
    assert store.is_cancel_requested("s1") is False


def test_request_cancel_works_on_non_terminal_status(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch)
    store.upsert_submitted("s1", "job-1", "/tmp/a.wav", 100.0, 1, now="t0")
    store.update_status("s1", "started", now="t1")
    ok = store.request_cancel("s1", now="t2")
    assert ok is True
    assert store.is_cancel_requested("s1") is True


def test_recover_orphaned_marks_only_non_terminal_rows(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch)
    store.upsert_submitted("running", "job-1", "/tmp/a.wav", 100.0, 1, now="t0")
    store.update_status("running", "started", now="t1")

    store.upsert_submitted("polling_job", "job-2", "/tmp/b.wav", 100.0, 1, now="t0")
    store.update_status("polling_job", "polling", now="t1")

    store.upsert_submitted("done", "job-3", "/tmp/c.wav", 100.0, 1, now="t0")
    store.update_status("done", "completed", now="t1")

    store.upsert_submitted("dead", "job-4", "/tmp/d.wav", 100.0, 1, now="t0")
    store.update_status("dead", "stt_failed", now="t1")

    recovered = store.recover_orphaned(now="t2")
    assert recovered == 2
    assert store.get("running")["status"] == "interrupted"
    assert store.get("polling_job")["status"] == "interrupted"
    assert store.get("done")["status"] == "completed"     # untouched
    assert store.get("dead")["status"] == "stt_failed"     # untouched


def test_stage_label_covers_every_status():
    for status in store.STATUSES:
        label = store.stage_label(status)
        assert isinstance(label, str) and label
    # Unknown status falls back to itself rather than raising/blank.
    assert store.stage_label("some_future_status") == "some_future_status"
