"""
tests/test_migration_and_resume_review.py
Targeted production-safety review for the Sarvam Batch STT segment-resume
work (commit 9c9b96e): migration safety against a pre-existing old-schema
DB, and the exact resume/concurrency/ordering/partial-failure scenarios
called out in that review. Complements (does not replace)
test_sarvam_job_store.py and test_pipeline_batch_stt.py.
"""
import json
import sqlite3
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))

import pipeline_v3
import transcription.sarvam_batch as sarvam_batch
import transcription.sarvam_job_store as sarvam_job_store
from audio.processor import AudioProcessor
from config.models import Speaker
from diarization.diarizer import Diarizer


def _fresh_job_store(tmp_path, monkeypatch):
    monkeypatch.setattr(sarvam_job_store, "DB_PATH", tmp_path / "sarvam_stt_jobs.db")
    monkeypatch.setattr(sarvam_job_store, "_local", threading.local())
    sarvam_job_store.init_db()


class _FakeSTT:
    def _translate_to_english(self, text, lang):
        return f"EN[{text}]"


def _pipeline_stub():
    import types
    stub = SimpleNamespace(p=None, processor=AudioProcessor(), stt=_FakeSTT(), diarizer=Diarizer())
    stub._process_segment = types.MethodType(pipeline_v3.DemoAuditPipelineV3._process_segment, stub)
    stub._utterances_from_result = types.MethodType(pipeline_v3.DemoAuditPipelineV3._utterances_from_result, stub)
    return stub


def _make_result(text, start, end, speaker_id="0"):
    return sarvam_batch.BatchTranscriptResult(
        transcript=text, language_code="en-IN", has_diarization=True,
        segments=[sarvam_batch.DiarizedSegment(text=text, start=start, end=end, speaker_id=speaker_id)],
    )


# ── 2. Migration safety: old-schema SQLite DB (mirrors what production had
# before this change) ────────────────────────────────────────────────────────

def test_migration_preserves_existing_row_from_old_schema(tmp_path, monkeypatch):
    db_path = tmp_path / "old.db"
    con = sqlite3.connect(str(db_path))
    con.executescript("""
    CREATE TABLE sarvam_stt_jobs (
        session_id       TEXT PRIMARY KEY,
        sarvam_job_id    TEXT DEFAULT '',
        status           TEXT NOT NULL DEFAULT 'submitted',
        wav_path         TEXT DEFAULT '',
        duration_seconds REAL,
        num_segments     INTEGER DEFAULT 1,
        retry_count      INTEGER DEFAULT 0,
        error_message    TEXT DEFAULT '',
        cancel_requested INTEGER NOT NULL DEFAULT 0,
        created_at       TEXT,
        updated_at       TEXT
    );
    """)
    con.execute("INSERT INTO sarvam_stt_jobs (session_id, sarvam_job_id, status, created_at, updated_at) "
                "VALUES ('existing-1', 'real-job-id', 'completed', 't0', 't0')")
    con.commit()
    con.close()

    monkeypatch.setattr(sarvam_job_store, "DB_PATH", db_path)
    monkeypatch.setattr(sarvam_job_store, "_local", threading.local())
    sarvam_job_store.init_db()

    row = sarvam_job_store.get("existing-1")
    assert row["sarvam_job_id"] == "real-job-id"
    assert row["status"] == "completed"
    assert row["parent_session_id"] == ""
    assert row["segment_index"] == 0
    assert row["transcript_json"] == ""


def test_migration_is_idempotent_across_repeated_init_db_calls(tmp_path, monkeypatch):
    _fresh_job_store(tmp_path, monkeypatch)
    sarvam_job_store.upsert_submitted("s1", "job-1", "/tmp/a.wav", 100.0, 1, now="t0")
    sarvam_job_store.save_transcript("s1", '{"x":1}', now="t1")
    before = sarvam_job_store.get("s1")

    for _ in range(3):
        monkeypatch.setattr(sarvam_job_store, "_local", threading.local())
        sarvam_job_store.init_db()

    after = sarvam_job_store.get("s1")
    assert dict(before) == dict(after)


# ── 3. Resume semantics — exact scenarios A/B/C/D ───────────────────────────

def test_resume_A_mixed_completed_processing_pending(tmp_path, monkeypatch):
    """Segment 1=COMPLETED, 2=COMPLETED, 3=PROCESSING, 4=PENDING. Resume must:
    never resend 1/2 to Sarvam, recover 3 appropriately, submit 4."""
    _fresh_job_store(tmp_path, monkeypatch)
    wav_paths = [tmp_path / f"seg{i}.wav" for i in range(4)]
    for p in wav_paths:
        p.write_bytes(b"x")

    sarvam_job_store.create_pending_segments("parent", [str(p) for p in wav_paths], now="t0")
    sarvam_job_store.upsert_submitted("parent::seg0", "job-0", str(wav_paths[0]), 10, 4, now="t0")
    sarvam_job_store.save_transcript("parent::seg0", json.dumps({"transcript": "a", "language_code": "en-IN",
                                     "has_diarization": True, "segments": [{"text": "a", "start": 0, "end": 1, "speaker_id": "0"}]}), now="t1")
    sarvam_job_store.upsert_submitted("parent::seg1", "job-1", str(wav_paths[1]), 10, 4, now="t0")
    sarvam_job_store.save_transcript("parent::seg1", json.dumps({"transcript": "b", "language_code": "en-IN",
                                     "has_diarization": True, "segments": [{"text": "b", "start": 0, "end": 1, "speaker_id": "0"}]}), now="t1")
    sarvam_job_store.upsert_submitted("parent::seg2", "job-2", str(wav_paths[2]), 10, 4, now="t0")
    sarvam_job_store.update_status("parent::seg2", "started", now="t0")   # PROCESSING when the crash happened
    # segment 3 stays 'pending' (never attempted yet)

    resolve_calls = []
    submit_calls = []
    monkeypatch.setattr(sarvam_batch, "resolve_existing_job_state",
                         lambda job_id: resolve_calls.append(job_id) or "Completed")
    monkeypatch.setattr(sarvam_batch, "get_job_handle", lambda job_id: SimpleNamespace(job_id=job_id))
    monkeypatch.setattr(sarvam_batch, "submit_job", lambda path: submit_calls.append(path) or SimpleNamespace(job_id="new-job"))
    monkeypatch.setattr(sarvam_batch, "upload_and_start", lambda job, path: None)
    monkeypatch.setattr(sarvam_batch, "poll_with_backoff", lambda job, sid, on_status=None, is_cancelled=None: "Completed")
    monkeypatch.setattr(sarvam_batch, "download_and_parse", lambda job, path, out_dir: _make_result("resumed", 0, 1))

    utterances = pipeline_v3.DemoAuditPipelineV3._batch_stt(_pipeline_stub(), "parent", list(zip(wav_paths, [0, 100, 200, 300])))

    # 1 and 2: cached transcript short-circuit — never even a status check.
    assert "job-0" not in resolve_calls
    assert "job-1" not in resolve_calls
    # 3: was PROCESSING — recovered via a live status check on its existing job, not resubmitted.
    assert "job-2" in resolve_calls
    # 4: had no prior job at all — must be freshly submitted.
    assert submit_calls == [wav_paths[3]]
    assert len(utterances) == 4
    native_texts = {u.native_text for u in utterances}
    assert "a" in native_texts and "b" in native_texts   # cached segments' real text preserved


def test_resume_B_all_completed_zero_sarvam_calls(tmp_path, monkeypatch):
    """All segments COMPLETED -> resume must make ZERO Sarvam API calls."""
    _fresh_job_store(tmp_path, monkeypatch)
    wav_paths = [tmp_path / f"seg{i}.wav" for i in range(3)]
    for p in wav_paths:
        p.write_bytes(b"x")
    sarvam_job_store.create_pending_segments("parent", [str(p) for p in wav_paths], now="t0")
    for i in range(3):
        sid = f"parent::seg{i}"
        sarvam_job_store.upsert_submitted(sid, f"job-{i}", str(wav_paths[i]), 10, 3, now="t0")
        sarvam_job_store.save_transcript(sid, json.dumps({"transcript": f"t{i}", "language_code": "en-IN",
                                          "has_diarization": True, "segments": [{"text": f"t{i}", "start": 0, "end": 1, "speaker_id": "0"}]}), now="t1")

    any_sarvam_call = []
    for name in ("submit_job", "upload_and_start", "resolve_existing_job_state", "get_job_handle",
                 "poll_with_backoff", "download_and_parse"):
        monkeypatch.setattr(sarvam_batch, name,
                             lambda *a, _n=name, **k: any_sarvam_call.append(_n) or SimpleNamespace(job_id="x"))

    utterances = pipeline_v3.DemoAuditPipelineV3._batch_stt(
        _pipeline_stub(), "parent", list(zip(wav_paths, [0, 100, 200])))

    assert any_sarvam_call == [], f"expected zero Sarvam calls, got: {any_sarvam_call}"
    assert len(utterances) == 3


def test_resume_C_one_failed_others_completed_only_failed_retried(tmp_path, monkeypatch):
    """One segment FAILED, others COMPLETED — completed stay cached and are
    never resubmitted; a subsequent retry only touches the failed one."""
    _fresh_job_store(tmp_path, monkeypatch)
    wav_paths = [tmp_path / f"seg{i}.wav" for i in range(3)]
    for p in wav_paths:
        p.write_bytes(b"x")
    sarvam_job_store.create_pending_segments("parent", [str(p) for p in wav_paths], now="t0")
    for i in (0, 2):
        sid = f"parent::seg{i}"
        sarvam_job_store.upsert_submitted(sid, f"job-{i}", str(wav_paths[i]), 10, 3, now="t0")
        sarvam_job_store.save_transcript(sid, json.dumps({"transcript": f"t{i}", "language_code": "en-IN",
                                          "has_diarization": True, "segments": [{"text": f"t{i}", "start": 0, "end": 1, "speaker_id": "0"}]}), now="t1")
    sarvam_job_store.upsert_submitted("parent::seg1", "job-1", str(wav_paths[1]), 10, 3, now="t0")
    sarvam_job_store.update_status("parent::seg1", "stt_failed", "boom", now="t0")

    submit_calls = []
    monkeypatch.setattr(sarvam_batch, "submit_job", lambda path: submit_calls.append(path) or SimpleNamespace(job_id="retry-job"))
    monkeypatch.setattr(sarvam_batch, "upload_and_start", lambda job, path: None)
    monkeypatch.setattr(sarvam_batch, "poll_with_backoff", lambda job, sid, on_status=None, is_cancelled=None: "Completed")
    monkeypatch.setattr(sarvam_batch, "download_and_parse", lambda job, path, out_dir: _make_result("recovered", 0, 1))

    utterances = pipeline_v3.DemoAuditPipelineV3._batch_stt(
        _pipeline_stub(), "parent", list(zip(wav_paths, [0, 100, 200])))

    # Only segment 1 (the previously failed one) was ever submitted to Sarvam.
    assert submit_calls == [wav_paths[1]]
    progress = sarvam_job_store.segment_progress("parent")
    assert progress["completed"] == 3
    assert progress["failed"] == 0


def test_resume_D_server_died_mid_flight_recovers_job_id_not_duplicate(tmp_path, monkeypatch):
    """Server/process dies while segments are 'started'/'polling' — startup
    recovery marks them 'interrupted'; the NEXT resume must recover the
    EXISTING Sarvam job_id rather than submitting a brand-new one."""
    _fresh_job_store(tmp_path, monkeypatch)
    wav_paths = [tmp_path / f"seg{i}.wav" for i in range(2)]
    for p in wav_paths:
        p.write_bytes(b"x")
    sarvam_job_store.create_pending_segments("parent", [str(p) for p in wav_paths], now="t0")
    sarvam_job_store.upsert_submitted("parent::seg0", "job-alive-on-sarvam", str(wav_paths[0]), 10, 2, now="t0")
    sarvam_job_store.update_status("parent::seg0", "started", now="t0")   # mid-flight when it died
    sarvam_job_store.upsert_submitted("parent::seg1", "job-1", str(wav_paths[1]), 10, 2, now="t0")
    sarvam_job_store.save_transcript("parent::seg1", json.dumps({"transcript": "b", "language_code": "en-IN",
                                      "has_diarization": True, "segments": [{"text": "b", "start": 0, "end": 1, "speaker_id": "0"}]}), now="t1")

    # Startup recovery runs (server restart).
    recovered = sarvam_job_store.recover_orphaned(now="t2")
    assert recovered == 1
    assert sarvam_job_store.get("parent::seg0")["status"] == "interrupted"
    assert sarvam_job_store.get("parent::seg0")["sarvam_job_id"] == "job-alive-on-sarvam"  # job_id NOT lost

    submit_calls = []
    monkeypatch.setattr(sarvam_batch, "submit_job", lambda path: submit_calls.append(path) or SimpleNamespace(job_id="brand-new-job"))
    monkeypatch.setattr(sarvam_batch, "resolve_existing_job_state", lambda job_id: "Completed")  # it finished on Sarvam's side while we were down
    monkeypatch.setattr(sarvam_batch, "get_job_handle", lambda job_id: SimpleNamespace(job_id=job_id))
    monkeypatch.setattr(sarvam_batch, "poll_with_backoff", lambda job, sid, on_status=None, is_cancelled=None: "Completed")
    monkeypatch.setattr(sarvam_batch, "download_and_parse", lambda job, path, out_dir: _make_result("recovered-after-restart", 0, 1))

    pipeline_v3.DemoAuditPipelineV3._batch_stt(_pipeline_stub(), "parent", list(zip(wav_paths, [0, 100])))

    assert submit_calls == [], "must recover the existing job_id, not create a duplicate Sarvam job"
    assert sarvam_job_store.get("parent::seg0")["sarvam_job_id"] == "job-alive-on-sarvam"


# ── 4. Concurrency: N does not determine the concurrency ceiling ───────────

def test_concurrency_ceiling_never_exceeded_regardless_of_N(tmp_path, monkeypatch):
    """Verifies the actual live concurrent-worker count never exceeds
    SARVAM_BATCH_SEGMENT_CONCURRENCY, tested at several N — including N far
    larger than the concurrency cap."""
    from config.settings import get_settings
    settings = get_settings()
    concurrency_cap = 3
    monkeypatch.setattr(settings, "sarvam_batch_segment_concurrency", concurrency_cap)

    for n in (2, 3, 10, 100):
        sub = tmp_path / f"n{n}"
        sub.mkdir(exist_ok=True)
        _fresh_job_store(sub, monkeypatch)
        wav_paths = [sub / f"seg{i}.wav" for i in range(n)]
        for p in wav_paths:
            p.write_bytes(b"x")

        active = {"count": 0, "max_seen": 0}
        lock = threading.Lock()

        def _fake_poll(job, sid, on_status=None, is_cancelled=None):
            with lock:
                active["count"] += 1
                active["max_seen"] = max(active["max_seen"], active["count"])
            time.sleep(0.02)   # hold the "slot" briefly so overlap is observable
            with lock:
                active["count"] -= 1
            return "Completed"

        monkeypatch.setattr(sarvam_batch, "submit_job", lambda path: SimpleNamespace(job_id=f"job-{path.stem}"))
        monkeypatch.setattr(sarvam_batch, "upload_and_start", lambda job, path: None)
        monkeypatch.setattr(sarvam_batch, "poll_with_backoff", _fake_poll)
        monkeypatch.setattr(sarvam_batch, "download_and_parse", lambda job, path, out_dir: _make_result("x", 0, 1))

        utterances = pipeline_v3.DemoAuditPipelineV3._batch_stt(
            _pipeline_stub(), "parent", list(zip(wav_paths, [float(i) for i in range(n)])))

        assert len(utterances) == n, f"N={n}: expected {n} utterances, got {len(utterances)}"
        assert active["max_seen"] <= concurrency_cap, (
            f"N={n}: concurrency ceiling violated — saw {active['max_seen']} simultaneous "
            f"workers, cap is {concurrency_cap}")


def test_batch_stt_handles_N_1000_segments_correctly(tmp_path, monkeypatch):
    """N=1000 is an extreme (illustrative) case — correctness at scale, not
    timing: proves nothing assumes a small/bounded N anywhere in the actual
    pipeline path (vs. test_sarvam_job_store.py's store-layer-only N=1000
    test, and the timed concurrency-ceiling test above capped at N=100 to
    keep the suite fast)."""
    _fresh_job_store(tmp_path, monkeypatch)
    n = 1000
    wav_dir = tmp_path / "many"
    wav_dir.mkdir()
    wav_paths = [wav_dir / f"seg{i}.wav" for i in range(n)]
    for p in wav_paths:
        p.write_bytes(b"x")

    monkeypatch.setattr(sarvam_batch, "submit_job", lambda path: SimpleNamespace(job_id=f"job-{path.stem}"))
    monkeypatch.setattr(sarvam_batch, "upload_and_start", lambda job, path: None)
    monkeypatch.setattr(sarvam_batch, "poll_with_backoff", lambda job, sid, on_status=None, is_cancelled=None: "Completed")
    monkeypatch.setattr(sarvam_batch, "download_and_parse",
                         lambda job, path, out_dir: _make_result(path.stem, 0, 1))

    utterances = pipeline_v3.DemoAuditPipelineV3._batch_stt(
        _pipeline_stub(), "parent", [(wav_paths[i], float(i)) for i in range(n)])

    assert len(utterances) == n
    progress = sarvam_job_store.segment_progress("parent")
    assert progress["total_segments"] == n
    assert progress["completed"] == n


# ── 5. Out-of-order completion: 4 -> 1 -> 6 -> 2 -> 5 -> 3 ──────────────────

def test_out_of_order_completion_progress_and_ordering_correct(tmp_path, monkeypatch):
    _fresh_job_store(tmp_path, monkeypatch)
    n = 6
    wav_paths = [tmp_path / f"seg{i}.wav" for i in range(n)]
    for p in wav_paths:
        p.write_bytes(b"x")

    # Segment i's fake Sarvam job takes longer the EARLIER it is in this
    # completion order, so real completion order is 4,1,6,2,5,3 (1-indexed
    # in the spec) == indices 3,0,5,1,4,2 (0-indexed) here.
    completion_order_0indexed = [3, 0, 5, 1, 4, 2]
    delay_for_index = {idx: (len(completion_order_0indexed) - pos) * 0.01
                        for pos, idx in enumerate(completion_order_0indexed)}

    def _fake_poll(job, sid, on_status=None, is_cancelled=None):
        idx = int(sid.split("seg")[1])
        time.sleep(delay_for_index[idx])
        return "Completed"

    monkeypatch.setattr(sarvam_batch, "submit_job", lambda path: SimpleNamespace(job_id=f"job-{path.stem}"))
    monkeypatch.setattr(sarvam_batch, "upload_and_start", lambda job, path: None)
    monkeypatch.setattr(sarvam_batch, "poll_with_backoff", _fake_poll)
    monkeypatch.setattr(sarvam_batch, "download_and_parse",
                         lambda job, path, out_dir: _make_result(path.stem, 0, 1))

    utterances = pipeline_v3.DemoAuditPipelineV3._batch_stt(
        _pipeline_stub(), "parent", [(wav_paths[i], float(i) * 1000) for i in range(n)])

    # Final utterances must be in CHRONOLOGICAL (offset) order regardless of
    # completion order.
    assert [u.native_text for u in utterances] == [f"seg{i}" for i in range(n)]

    # Progress must reflect real persisted COUNT — segment 6 (index 5)
    # finishing does not imply segments 1-5 are complete; by the time
    # _batch_stt returns, ALL must show completed (COUNT-based, not
    # "highest index").
    progress = sarvam_job_store.segment_progress("parent")
    assert progress["total_segments"] == 6
    assert progress["completed"] == 6
    assert progress["pending"] == 0
    assert progress["failed"] == 0


# ── 6. Partial failure: 10 segments, 8 successful, 2 failed ─────────────────

def test_partial_failure_10_segments_8_success_2_failed(tmp_path, monkeypatch):
    _fresh_job_store(tmp_path, monkeypatch)
    n = 10
    wav_paths = [tmp_path / f"seg{i}.wav" for i in range(n)]
    for p in wav_paths:
        p.write_bytes(b"x")
    failed_indices = {2, 7}

    def _fake_poll(job, sid, on_status=None, is_cancelled=None):
        idx = int(sid.split("seg")[1])
        return "Failed" if idx in failed_indices else "Completed"

    def _fake_submit(path):
        idx = int(path.stem.replace("seg", ""))
        h = SimpleNamespace(job_id=f"job-{idx}")
        h.get_file_results = lambda: {"failed": [{"error_message": f"segment {idx} bad audio"}]}
        return h

    monkeypatch.setattr(sarvam_batch, "submit_job", _fake_submit)
    monkeypatch.setattr(sarvam_batch, "upload_and_start", lambda job, path: None)
    monkeypatch.setattr(sarvam_batch, "poll_with_backoff", _fake_poll)
    monkeypatch.setattr(sarvam_batch, "download_and_parse",
                         lambda job, path, out_dir: _make_result(path.stem, 0, 1))

    try:
        pipeline_v3.DemoAuditPipelineV3._batch_stt(
            _pipeline_stub(), "parent", [(wav_paths[i], float(i) * 100) for i in range(n)])
        assert False, "expected SttJobFailedError"
    except sarvam_batch.SttJobFailedError as exc:
        assert "2/10 segment(s) failed" in str(exc)

    progress = sarvam_job_store.segment_progress("parent")
    assert progress["completed"] == 8
    assert progress["failed"] == 2
    # The 8 successful segments' transcripts are cached — a subsequent retry
    # would resubmit only the 2 failed ones (proven by test_resume_C above
    # using the same cache mechanism at smaller scale).
    for i in range(n):
        row = sarvam_job_store.get(f"parent::seg{i}")
        if i in failed_indices:
            assert row["status"] == "stt_failed"
            assert row["transcript_json"] == ""
        else:
            assert row["status"] == "completed"
            assert row["transcript_json"] != ""


# ── 7. Stale sweep: recently-heartbeating segment must NOT be swept ─────────

def test_stale_sweep_does_not_touch_actively_heartbeating_segment(tmp_path, monkeypatch):
    _fresh_job_store(tmp_path, monkeypatch)
    sarvam_job_store.upsert_submitted("s1", "job-1", "/tmp/a.wav", 10, 1, now="2020-01-01T00:00:00")
    sarvam_job_store.update_status("s1", "polling", now="2020-01-01T00:09:55")  # heartbeat 5s ago

    recovered = sarvam_job_store.mark_stale_as_interrupted(
        threshold_seconds=600, now="2020-01-01T00:10:00")   # 10-min threshold, only 5s since last heartbeat

    assert recovered == 0
    assert sarvam_job_store.get("s1")["status"] == "polling"
