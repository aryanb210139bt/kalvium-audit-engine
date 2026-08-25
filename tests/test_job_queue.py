"""
tests/test_job_queue.py
Unit tests for the upload -> queue -> manual start workflow's persistence
layer (job_queue.py). Covers the core guarantee this feature exists for:
uploading never itself starts processing, and the queue survives being
read back fresh (i.e. would survive a server restart, since it's SQLite-
backed rather than in-memory).
"""
import json
import sys
import threading
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import job_queue


def _fresh_db(tmp_path, monkeypatch, name):
    monkeypatch.setattr(job_queue, "DB_PATH", tmp_path / name)
    monkeypatch.setattr(job_queue, "_local", threading.local())
    job_queue.init_db()


def test_enqueue_creates_queued_job_not_processing(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch, "q1.db")
    job_id = str(uuid.uuid4())
    job_queue.enqueue(job_id, "url", label="Test", actor="pytest",
                       payload={"url": "https://x"}, created_at="2026-01-01T00:00:00")
    job = job_queue.get(job_id)
    assert job["status"] == "queued"
    assert job["payload"]["url"] == "https://x"


def test_list_active_excludes_terminal_states(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch, "q2.db")
    q = str(uuid.uuid4()); p = str(uuid.uuid4()); c = str(uuid.uuid4()); f = str(uuid.uuid4())
    for jid in (q, p, c, f):
        job_queue.enqueue(jid, "url", label=jid[:4], actor="pytest", payload={}, created_at="2026-01-01T00:00:00")
    job_queue.mark_processing(p, "2026-01-01T00:00:01")
    job_queue.mark_processing(c, "2026-01-01T00:00:01")
    job_queue.mark_completed(c, "2026-01-01T00:00:02")
    job_queue.mark_processing(f, "2026-01-01T00:00:01")
    job_queue.mark_failed(f, "boom", "2026-01-01T00:00:02")

    active_ids = {j["job_id"] for j in job_queue.list_active()}
    assert q in active_ids
    assert p in active_ids
    assert c not in active_ids
    assert f not in active_ids


def test_mark_processing_only_after_starting(tmp_path, monkeypatch):
    # This is the mechanism "Start All" relies on: a job that's been told to
    # start but whose worker thread hasn't picked it up yet must NOT show as
    # 'processing' — only mark_processing() (called from inside the run
    # function itself) makes that transition.
    _fresh_db(tmp_path, monkeypatch, "q3.db")
    job_id = str(uuid.uuid4())
    job_queue.enqueue(job_id, "url", label="Test", actor="pytest", payload={}, created_at="2026-01-01T00:00:00")
    job_queue.mark_starting(job_id)
    assert job_queue.get(job_id)["status"] == "starting"  # not yet 'processing'
    job_queue.mark_processing(job_id, "2026-01-01T00:00:05")
    assert job_queue.get(job_id)["status"] == "processing"


def test_remove_only_works_on_queued(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch, "q4.db")
    queued_id = str(uuid.uuid4())
    processing_id = str(uuid.uuid4())
    job_queue.enqueue(queued_id, "url", label="Q", actor="pytest", payload={}, created_at="2026-01-01T00:00:00")
    job_queue.enqueue(processing_id, "url", label="P", actor="pytest", payload={}, created_at="2026-01-01T00:00:00")
    job_queue.mark_processing(processing_id, "2026-01-01T00:00:01")

    assert job_queue.remove(queued_id) is True
    assert job_queue.get(queued_id)["status"] == "removed"

    assert job_queue.remove(processing_id) is False  # spec: never silently delete a processing job
    assert job_queue.get(processing_id)["status"] == "processing"


def test_retry_only_works_on_failed(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch, "q5.db")
    job_id = str(uuid.uuid4())
    job_queue.enqueue(job_id, "url", label="Test", actor="pytest", payload={}, created_at="2026-01-01T00:00:00")
    job_queue.mark_processing(job_id, "2026-01-01T00:00:01")
    job_queue.mark_failed(job_id, "network error", "2026-01-01T00:00:02")

    assert job_queue.retry(job_id, "2026-01-01T00:01:00") is True
    job = job_queue.get(job_id)
    assert job["status"] == "queued"
    assert job["error"] is None

    # Can't retry something that's already queued (not failed)
    assert job_queue.retry(job_id, "2026-01-01T00:02:00") is False


def test_order_index_assigned_sequentially(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch, "q6.db")
    ids = [str(uuid.uuid4()) for _ in range(3)]
    for jid in ids:
        job_queue.enqueue(jid, "url", label="x", actor="pytest", payload={}, created_at="2026-01-01T00:00:00")
    orders = [job_queue.get(jid)["order_index"] for jid in ids]
    assert orders == sorted(orders)
    assert len(set(orders)) == 3  # each distinct


def test_reorder_moves_a_queued_job(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch, "q7.db")
    a, b = str(uuid.uuid4()), str(uuid.uuid4())
    job_queue.enqueue(a, "url", label="A", actor="pytest", payload={}, created_at="2026-01-01T00:00:00")
    job_queue.enqueue(b, "url", label="B", actor="pytest", payload={}, created_at="2026-01-01T00:00:00")
    job_queue.reorder(b, -1)
    ordered = [j["job_id"] for j in job_queue.list_queued_only()]
    assert ordered == [b, a]


def test_cancel_hides_a_processing_job(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch, "q10.db")
    job_id = str(uuid.uuid4())
    job_queue.enqueue(job_id, "url", label="Test", actor="pytest", payload={}, created_at="2026-01-01T00:00:00")
    job_queue.mark_processing(job_id, "2026-01-01T00:00:01")

    assert job_queue.cancel(job_id, "2026-01-01T00:05:00") is True
    job = job_queue.get(job_id)
    assert job["status"] == "cancelled"
    assert job["job_id"] not in {j["job_id"] for j in job_queue.list_active()}  # hidden from both lists


def test_cancel_fails_on_already_terminal_job(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch, "q11.db")
    job_id = str(uuid.uuid4())
    job_queue.enqueue(job_id, "url", label="Test", actor="pytest", payload={}, created_at="2026-01-01T00:00:00")
    job_queue.mark_processing(job_id, "2026-01-01T00:00:01")
    job_queue.mark_completed(job_id, "2026-01-01T00:00:02")

    assert job_queue.cancel(job_id, "2026-01-01T00:05:00") is False
    assert job_queue.get(job_id)["status"] == "completed"  # unchanged


def test_cancelled_job_stays_cancelled_even_if_background_finishes_later(tmp_path, monkeypatch):
    # The background thread isn't actually stopped by a soft cancel — it may
    # still call mark_completed/mark_failed once it eventually finishes.
    # Those calls must not resurrect a job the user already hid.
    _fresh_db(tmp_path, monkeypatch, "q12.db")
    job_id = str(uuid.uuid4())
    job_queue.enqueue(job_id, "url", label="Test", actor="pytest", payload={}, created_at="2026-01-01T00:00:00")
    job_queue.mark_processing(job_id, "2026-01-01T00:00:01")
    job_queue.cancel(job_id, "2026-01-01T00:02:00")

    job_queue.mark_completed(job_id, "2026-01-01T00:10:00")
    assert job_queue.get(job_id)["status"] == "cancelled"  # still cancelled, not resurrected

    job_queue.mark_failed(job_id, "some error", "2026-01-01T00:10:00")
    assert job_queue.get(job_id)["status"] == "cancelled"  # same for the failure path


def test_recover_orphaned_resets_starting_and_processing_to_queued(tmp_path, monkeypatch):
    # Regression test for a real incident: a server restart while jobs were
    # 'starting'/'processing' left them permanently stuck (no executor
    # submission survives a restart, but the DB row didn't know that) — no
    # Start button (already 'starting'), no Remove button either. This is
    # the fix: call recover_orphaned() once at startup.
    _fresh_db(tmp_path, monkeypatch, "q9.db")
    starting_id = str(uuid.uuid4())
    processing_id = str(uuid.uuid4())
    queued_id = str(uuid.uuid4())
    completed_id = str(uuid.uuid4())
    for jid in (starting_id, processing_id, queued_id, completed_id):
        job_queue.enqueue(jid, "url", label="x", actor="pytest", payload={}, created_at="2026-01-01T00:00:00")
    job_queue.mark_starting(starting_id)
    job_queue.mark_processing(processing_id, "2026-01-01T00:00:01")
    job_queue.mark_processing(completed_id, "2026-01-01T00:00:01")
    job_queue.mark_completed(completed_id, "2026-01-01T00:00:02")

    recovered = job_queue.recover_orphaned()
    assert recovered == 2  # starting_id + processing_id, not queued_id or completed_id

    assert job_queue.get(starting_id)["status"] == "queued"
    assert job_queue.get(starting_id)["started_at"] is None
    assert job_queue.get(processing_id)["status"] == "queued"
    assert job_queue.get(queued_id)["status"] == "queued"       # untouched
    assert job_queue.get(completed_id)["status"] == "completed"  # untouched — real result preserved


def test_survives_fresh_read_simulating_restart(tmp_path, monkeypatch):
    # No in-memory state is involved here at all — a fresh _conn() (as would
    # happen on a server restart) reads the same data straight from disk.
    _fresh_db(tmp_path, monkeypatch, "q8.db")
    job_id = str(uuid.uuid4())
    job_queue.enqueue(job_id, "file", label="Persisted", actor="pytest",
                       payload={"recording_path": "/tmp/x.mp4"}, created_at="2026-01-01T00:00:00",
                       filename="x.mp4")
    monkeypatch.setattr(job_queue, "_local", threading.local())  # simulate a new process's fresh connection
    job = job_queue.get(job_id)
    assert job is not None
    assert job["status"] == "queued"
    assert job["payload"]["recording_path"] == "/tmp/x.mp4"
