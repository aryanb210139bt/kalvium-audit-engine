"""
tests/test_stt_status_endpoint.py
Endpoint-level tests for GET /api/v1/audit/{session_id}/stt-status —
verifies backward compatibility of the response shape across every case:
single-job (pre-existing shape, unchanged), multi-segment (new,
aggregated), completed, partially-completed, failed, and unknown session.
Uses FastAPI's TestClient directly against the real app (conftest.py's
autouse fixture already forces DB_BACKEND=sqlite for the whole process).
"""
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import transcription.sarvam_job_store as sarvam_job_store


def _fresh_job_store(tmp_path, monkeypatch):
    monkeypatch.setattr(sarvam_job_store, "DB_PATH", tmp_path / "sarvam_stt_jobs.db")
    monkeypatch.setattr(sarvam_job_store, "_local", threading.local())
    sarvam_job_store.init_db()


def _client():
    from fastapi.testclient import TestClient
    import api.main as main
    return TestClient(main.app)


def test_unknown_session_returns_404(tmp_path, monkeypatch):
    _fresh_job_store(tmp_path, monkeypatch)
    resp = _client().get("/api/v1/audit/does-not-exist/stt-status")
    assert resp.status_code == 404
    assert "No Sarvam Batch STT job" in resp.json()["detail"]


def test_single_job_shape_unchanged_for_backward_compatibility(tmp_path, monkeypatch):
    """This is the PRE-EXISTING response shape (from the first Batch STT
    deploy) — any existing frontend/consumer code depending on these exact
    keys must keep working unchanged."""
    _fresh_job_store(tmp_path, monkeypatch)
    sarvam_job_store.upsert_submitted("s1", "job-abc", "/tmp/a.wav", 5400.0, 1, now="t0")
    sarvam_job_store.save_transcript("s1", '{"x":1}', now="t1")

    resp = _client().get("/api/v1/audit/s1/stt-status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == "s1"
    assert body["sarvam_job_id"] == "job-abc"
    assert body["status"] == "completed"
    assert body["stage"] == "Sarvam Batch STT → Completed — Transcript ready"
    assert body["retry_count"] == 0
    assert body["num_segments"] == 1
    assert body["error_message"] is None
    assert "updated_at" in body
    # These keys must NOT appear on the single-job shape (would indicate
    # accidental cross-contamination with the multi-segment shape).
    assert "total_segments" not in body
    assert "segments" not in body


def test_single_job_failed_shape(tmp_path, monkeypatch):
    _fresh_job_store(tmp_path, monkeypatch)
    sarvam_job_store.upsert_submitted("s1", "job-abc", "/tmp/a.wav", 100.0, 1, now="t0")
    sarvam_job_store.update_status("s1", "stt_failed", "bad audio codec", now="t1")

    body = _client().get("/api/v1/audit/s1/stt-status").json()
    assert body["status"] == "stt_failed"
    assert body["error_message"] == "bad audio codec"


def test_multi_segment_shape_aggregates_correctly(tmp_path, monkeypatch):
    _fresh_job_store(tmp_path, monkeypatch)
    wav_paths = [f"/tmp/seg{i}.wav" for i in range(4)]
    sarvam_job_store.create_pending_segments("parent-1", wav_paths, now="t0")
    sarvam_job_store.upsert_submitted("parent-1::seg0", "job-0", wav_paths[0], 10, 4, now="t0")
    sarvam_job_store.save_transcript("parent-1::seg0", '{"x":1}', now="t1")
    sarvam_job_store.upsert_submitted("parent-1::seg1", "job-1", wav_paths[1], 10, 4, now="t0")
    sarvam_job_store.update_status("parent-1::seg1", "started", now="t1")
    # segment 2 stays 'pending'; segment 3 fails
    sarvam_job_store.upsert_submitted("parent-1::seg3", "job-3", wav_paths[3], 10, 4, now="t0")
    sarvam_job_store.update_status("parent-1::seg3", "stt_failed", "boom", now="t1")

    body = _client().get("/api/v1/audit/parent-1/stt-status").json()
    assert body["session_id"] == "parent-1"
    assert body["total_segments"] == 4
    assert body["completed"] == 1
    assert body["processing"] == 1
    assert body["pending"] == 1   # seg2, never touched — distinct from "failed"
    assert body["failed"] == 1    # seg3 — its own bucket, not folded into pending
    assert body["status"] == "processing"   # not all done, not fully failed
    assert len(body["segments"]) == 4
    assert body["segments"][0]["segment_index"] == 0


def test_multi_segment_all_completed_status_is_completed(tmp_path, monkeypatch):
    _fresh_job_store(tmp_path, monkeypatch)
    wav_paths = [f"/tmp/seg{i}.wav" for i in range(2)]
    sarvam_job_store.create_pending_segments("parent-1", wav_paths, now="t0")
    for i in range(2):
        sid = f"parent-1::seg{i}"
        sarvam_job_store.upsert_submitted(sid, f"job-{i}", wav_paths[i], 10, 2, now="t0")
        sarvam_job_store.save_transcript(sid, '{"x":1}', now="t1")

    body = _client().get("/api/v1/audit/parent-1/stt-status").json()
    assert body["status"] == "completed"
    assert body["completed"] == 2
    assert body["progress_pct"] == 100.0


def test_multi_segment_all_failed_status_is_stt_failed(tmp_path, monkeypatch):
    _fresh_job_store(tmp_path, monkeypatch)
    wav_paths = [f"/tmp/seg{i}.wav" for i in range(2)]
    sarvam_job_store.create_pending_segments("parent-1", wav_paths, now="t0")
    for i in range(2):
        sid = f"parent-1::seg{i}"
        sarvam_job_store.upsert_submitted(sid, f"job-{i}", wav_paths[i], 10, 2, now="t0")
        sarvam_job_store.update_status(sid, "stt_failed", "boom", now="t1")

    body = _client().get("/api/v1/audit/parent-1/stt-status").json()
    assert body["status"] == "stt_failed"


def test_multi_segment_endpoint_sweeps_stale_segments(tmp_path, monkeypatch):
    """The endpoint opportunistically sweeps stale segments before reporting
    — a segment stuck 'started' past the threshold should show as
    'interrupted' (folded into 'pending') on the very next status check,
    without needing a separate server restart."""
    _fresh_job_store(tmp_path, monkeypatch)
    from config.settings import get_settings
    monkeypatch.setattr(get_settings(), "sarvam_batch_stale_threshold_sec", 300)

    wav_paths = [f"/tmp/seg{i}.wav" for i in range(1)]
    sarvam_job_store.create_pending_segments("parent-1", wav_paths, now="2020-01-01T00:00:00")
    sarvam_job_store.upsert_submitted("parent-1::seg0", "job-0", wav_paths[0], 10, 1, now="2020-01-01T00:00:00")
    sarvam_job_store.update_status("parent-1::seg0", "started", now="2020-01-01T00:00:00")  # long-stale heartbeat

    resp = _client().get("/api/v1/audit/parent-1/stt-status")
    # The sweep uses the real current time, so a heartbeat from 2020 is
    # certainly stale under any reasonable threshold — must be recovered.
    assert sarvam_job_store.get("parent-1::seg0")["status"] == "interrupted"
    assert resp.status_code == 200
