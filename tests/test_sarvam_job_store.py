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


# ── Multi-segment (>2h recording, N derived dynamically) ────────────────────

def test_create_pending_segments_creates_one_row_per_wav_dynamically(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch)
    # N=5 here is arbitrary/illustrative — nothing in the implementation may
    # assume any particular N; the same code path must work for N=2 or N=500.
    wav_paths = [f"/tmp/seg{i}.wav" for i in range(5)]
    store.create_pending_segments("parent-1", wav_paths, now="t0")

    segments = store.list_segments("parent-1")
    assert len(segments) == 5
    assert [s["segment_index"] for s in segments] == [0, 1, 2, 3, 4]
    assert all(s["status"] == "pending" for s in segments)
    assert all(s["parent_session_id"] == "parent-1" for s in segments)
    assert [s["session_id"] for s in segments] == [f"parent-1::seg{i}" for i in range(5)]


def test_create_pending_segments_is_idempotent_never_resets_progress(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch)
    wav_paths = [f"/tmp/seg{i}.wav" for i in range(3)]
    store.create_pending_segments("parent-1", wav_paths, now="t0")

    # Segment 1 has since progressed to completed — a resume re-calling
    # create_pending_segments (idempotent pre-creation) must NOT reset it.
    store.upsert_submitted("parent-1::seg1", "job-1", "/tmp/seg1.wav", 100.0, 3, now="t1")
    store.save_transcript("parent-1::seg1", '{"transcript":"hi"}', now="t2")

    store.create_pending_segments("parent-1", wav_paths, now="t3")

    segments = {s["segment_index"]: s for s in store.list_segments("parent-1")}
    assert segments[0]["status"] == "pending"
    assert segments[1]["status"] == "completed"   # untouched, not reset
    assert segments[2]["status"] == "pending"


def test_segment_progress_is_count_based_not_last_index(tmp_path, monkeypatch):
    """Mirrors the exact out-of-order scenario from the spec:
    1=completed, 2=completed, 3=processing, 4=completed, 5=failed, 6=completed.
    Progress must reflect real COUNT(status=X), not "chunk 6 done so 6 total done"."""
    _fresh_db(tmp_path, monkeypatch)
    wav_paths = [f"/tmp/seg{i}.wav" for i in range(6)]
    store.create_pending_segments("parent-1", wav_paths, now="t0")

    plan = {0: "completed", 1: "completed", 2: "started", 3: "completed", 4: "stt_failed", 5: "completed"}
    for i, status in plan.items():
        sid = f"parent-1::seg{i}"
        store.upsert_submitted(sid, f"job-{i}", wav_paths[i], 100.0, 6, now="t1")
        if status == "completed":
            store.save_transcript(sid, "{}", now="t2")
        else:
            store.update_status(sid, status, now="t2")

    progress = store.segment_progress("parent-1")
    assert progress["total_segments"] == 6
    assert progress["completed"] == 4     # segments 0,1,3,5 — not "up to segment 6"
    assert progress["processing"] == 1    # segment 2
    assert progress["failed"] == 1        # segment 4
    assert progress["progress_pct"] == round(4 / 6 * 100, 1)


def test_segment_progress_dynamic_N_10_and_1000(tmp_path, monkeypatch):
    """The same aggregation logic must work whether N is small or large —
    nothing may assume a particular scale."""
    for n in (10, 1000):
        sub = tmp_path / str(n)
        sub.mkdir()
        _fresh_db(sub, monkeypatch)
        wav_paths = [f"/tmp/seg{i}.wav" for i in range(n)]
        store.create_pending_segments("parent", wav_paths, now="t0")
        progress = store.segment_progress("parent")
        assert progress["total_segments"] == n
        assert progress["pending"] == n
        assert progress["completed"] == 0


def test_save_transcript_caches_payload_and_marks_completed(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch)
    store.upsert_submitted("s1", "job-1", "/tmp/a.wav", 100.0, 1, now="t0")
    store.save_transcript("s1", '{"transcript": "hello"}', now="t1")
    row = store.get("s1")
    assert row["status"] == "completed"
    assert row["transcript_json"] == '{"transcript": "hello"}'


def test_mark_stale_as_interrupted_only_touches_old_non_terminal_rows(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch)
    old_time = "2020-01-01T00:00:00"
    recent_time = "2020-01-01T00:09:00"   # 9 minutes later — under a 600s(=10min) threshold... use tighter below

    store.upsert_submitted("stale", "job-1", "/tmp/a.wav", 100.0, 1, now=old_time)
    store.update_status("stale", "started", now=old_time)

    store.upsert_submitted("fresh", "job-2", "/tmp/b.wav", 100.0, 1, now=old_time)
    store.update_status("fresh", "started", now="2020-01-01T00:09:50")   # updated recently

    store.upsert_submitted("done", "job-3", "/tmp/c.wav", 100.0, 1, now=old_time)
    store.save_transcript("done", "{}", now=old_time)   # terminal — never touched

    now = "2020-01-01T00:10:00"   # 10 minutes after old_time
    recovered = store.mark_stale_as_interrupted(threshold_seconds=300, now=now)  # 5-min threshold

    assert recovered == 1
    assert store.get("stale")["status"] == "interrupted"
    assert store.get("fresh")["status"] == "started"      # too recent to be stale
    assert store.get("done")["status"] == "completed"     # terminal, untouched
