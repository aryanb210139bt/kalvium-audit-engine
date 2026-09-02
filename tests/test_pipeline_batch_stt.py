"""
tests/test_pipeline_batch_stt.py
Unit tests for DemoAuditPipelineV3._batch_stt — the orchestration glue
between transcription/sarvam_batch.py (mocked here) and
transcription/sarvam_job_store.py (real, pointed at a temp DB). Focuses on
the two things that matter most for correctness:
  1. A fresh run builds the right Utterances (timestamps, speaker roles
     from Sarvam's diarized_transcript via the existing talk-time
     heuristic, translated english_text).
  2. Smart-resume: an already-Completed prior job is never resubmitted —
     this is the actual fix for "retry must not re-pay for a finished
     Sarvam job".

Does NOT construct a real DemoAuditPipelineV3() (its __init__ loads heavy
ML models, e.g. XLM-RoBERTa sentiment) — _batch_stt only touches
self.p/self.processor/self.stt/self.diarizer, so a lightweight stand-in
with just those four attributes is enough and keeps this test fast.
"""
import json
import sys
import threading
import types
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
    """
    A lightweight stand-in for DemoAuditPipelineV3 (avoids constructing the
    real class, whose __init__ loads heavy ML models). _batch_stt now
    delegates to two sibling methods (_process_segment,
    _utterances_from_result) via `self.` — bind them onto the stub too via
    types.MethodType so `self._process_segment(...)` resolves the same way
    it would on a real instance.
    """
    stub = SimpleNamespace(p=None, processor=AudioProcessor(), stt=_FakeSTT(), diarizer=Diarizer())
    stub._process_segment = types.MethodType(pipeline_v3.DemoAuditPipelineV3._process_segment, stub)
    stub._utterances_from_result = types.MethodType(pipeline_v3.DemoAuditPipelineV3._utterances_from_result, stub)
    return stub


class _FakeJobHandle:
    def __init__(self, job_id):
        self.job_id = job_id


def test_batch_stt_fresh_submit_builds_utterances_with_roles(tmp_path, monkeypatch):
    _fresh_job_store(tmp_path, monkeypatch)
    wav_path = tmp_path / "call.wav"
    wav_path.write_bytes(b"not-a-real-wav")

    submitted = {}

    def _fake_submit_job(path):
        submitted["path"] = path
        return _FakeJobHandle("job-1")

    monkeypatch.setattr(sarvam_batch, "submit_job", _fake_submit_job)
    monkeypatch.setattr(sarvam_batch, "upload_and_start",
                         lambda job, path: submitted.setdefault("uploaded", path))
    monkeypatch.setattr(sarvam_batch, "poll_with_backoff",
                         lambda job, sid, on_status=None, is_cancelled=None:
                             (on_status(SimpleNamespace(job_id=job.job_id, job_state="Completed")) or "Completed")
                             if on_status else "Completed")
    monkeypatch.setattr(sarvam_batch, "download_and_parse", lambda job, path, out_dir: sarvam_batch.BatchTranscriptResult(
        transcript="hi", language_code="hi-IN", has_diarization=True,
        segments=[
            sarvam_batch.DiarizedSegment(text="namaste", start=0.0, end=10.0, speaker_id="0"),
            sarvam_batch.DiarizedSegment(text="haan bataiye", start=10.0, end=12.0, speaker_id="1"),
            sarvam_batch.DiarizedSegment(text="theek hai", start=12.0, end=20.0, speaker_id="0"),
        ],
    ))

    utterances = pipeline_v3.DemoAuditPipelineV3._batch_stt(_pipeline_stub(), "session-1", [(wav_path, 0.0)])

    assert submitted["path"] == wav_path
    assert len(utterances) == 3
    # speaker "0" talks 10s + 8s = 18s total vs speaker "1"'s 2s -> "0"=Counsellor, "1"=Student
    assert utterances[0].speaker == Speaker.COUNSELLOR
    assert utterances[1].speaker == Speaker.STUDENT
    assert utterances[2].speaker == Speaker.COUNSELLOR
    assert utterances[0].english_text == "EN[namaste]"
    assert utterances[0].start_time == 0.0 and utterances[0].end_time == 10.0

    job_row = sarvam_job_store.get("session-1")
    assert job_row["status"] == "completed"
    assert job_row["sarvam_job_id"] == "job-1"


def test_batch_stt_applies_time_offset_for_split_segments(tmp_path, monkeypatch):
    _fresh_job_store(tmp_path, monkeypatch)
    wav_path = tmp_path / "seg1.wav"
    wav_path.write_bytes(b"x")

    monkeypatch.setattr(sarvam_batch, "submit_job", lambda path: _FakeJobHandle("job-1"))
    monkeypatch.setattr(sarvam_batch, "upload_and_start", lambda job, path: None)
    monkeypatch.setattr(sarvam_batch, "poll_with_backoff",
                         lambda job, sid, on_status=None, is_cancelled=None: "Completed")
    monkeypatch.setattr(sarvam_batch, "download_and_parse", lambda job, path, out_dir: sarvam_batch.BatchTranscriptResult(
        transcript="hi", language_code="en-IN", has_diarization=True,
        segments=[sarvam_batch.DiarizedSegment(text="hello", start=1.0, end=2.0, speaker_id="0")],
    ))

    # This segment starts at offset=7000s within the full recording (e.g.
    # the second half of a >2h recording split by split_for_batch).
    utterances = pipeline_v3.DemoAuditPipelineV3._batch_stt(_pipeline_stub(), "session-1", [(wav_path, 7000.0)])

    assert utterances[0].start_time == 7001.0
    assert utterances[0].end_time == 7002.0


def test_batch_stt_skips_resubmission_when_prior_job_already_completed(tmp_path, monkeypatch):
    _fresh_job_store(tmp_path, monkeypatch)
    wav_path = tmp_path / "call.wav"
    wav_path.write_bytes(b"x")

    sarvam_job_store.upsert_submitted("session-1", "old-job-99", str(wav_path), 100.0, 1, now="t0")
    sarvam_job_store.update_status("session-1", "started", now="t1")

    submit_called = []
    monkeypatch.setattr(sarvam_batch, "submit_job", lambda path: submit_called.append(path))
    monkeypatch.setattr(sarvam_batch, "resolve_existing_job_state", lambda job_id: "Completed")
    monkeypatch.setattr(sarvam_batch, "get_job_handle", lambda job_id: _FakeJobHandle(job_id))
    monkeypatch.setattr(sarvam_batch, "poll_with_backoff",
                         lambda job, sid, on_status=None, is_cancelled=None: "Completed")
    monkeypatch.setattr(sarvam_batch, "download_and_parse", lambda job, path, out_dir: sarvam_batch.BatchTranscriptResult(
        transcript="x", language_code="en-IN", has_diarization=False, segments=[],
    ))

    pipeline_v3.DemoAuditPipelineV3._batch_stt(_pipeline_stub(), "session-1", [(wav_path, 0.0)])

    assert submit_called == []   # never resubmitted — the whole point of the fix
    row = sarvam_job_store.get("session-1")
    assert row["sarvam_job_id"] == "old-job-99"   # unchanged
    assert row["status"] == "completed"


def test_batch_stt_resumes_polling_when_prior_job_still_running(tmp_path, monkeypatch):
    _fresh_job_store(tmp_path, monkeypatch)
    wav_path = tmp_path / "call.wav"
    wav_path.write_bytes(b"x")

    sarvam_job_store.upsert_submitted("session-1", "old-job-99", str(wav_path), 100.0, 1, now="t0")
    sarvam_job_store.update_status("session-1", "interrupted", now="t1")  # e.g. after a server restart

    submit_called = []
    upload_called = []
    monkeypatch.setattr(sarvam_batch, "submit_job", lambda path: submit_called.append(path))
    monkeypatch.setattr(sarvam_batch, "upload_and_start", lambda job, path: upload_called.append(path))
    monkeypatch.setattr(sarvam_batch, "resolve_existing_job_state", lambda job_id: "Running")
    monkeypatch.setattr(sarvam_batch, "get_job_handle", lambda job_id: _FakeJobHandle(job_id))
    monkeypatch.setattr(sarvam_batch, "poll_with_backoff",
                         lambda job, sid, on_status=None, is_cancelled=None: "Completed")
    monkeypatch.setattr(sarvam_batch, "download_and_parse", lambda job, path, out_dir: sarvam_batch.BatchTranscriptResult(
        transcript="x", language_code="en-IN", has_diarization=False, segments=[],
    ))

    pipeline_v3.DemoAuditPipelineV3._batch_stt(_pipeline_stub(), "session-1", [(wav_path, 0.0)])

    assert submit_called == []
    assert upload_called == []   # neither initialise nor upload/start ran again
    assert sarvam_job_store.get("session-1")["sarvam_job_id"] == "old-job-99"


def test_batch_stt_raises_and_marks_stt_failed_on_terminal_failure(tmp_path, monkeypatch):
    _fresh_job_store(tmp_path, monkeypatch)
    wav_path = tmp_path / "call.wav"
    wav_path.write_bytes(b"x")

    monkeypatch.setattr(sarvam_batch, "submit_job", lambda path: _FakeJobHandle("job-1"))
    monkeypatch.setattr(sarvam_batch, "upload_and_start", lambda job, path: None)
    monkeypatch.setattr(sarvam_batch, "poll_with_backoff",
                         lambda job, sid, on_status=None, is_cancelled=None: "Failed")

    def _fake_get_file_results():
        return {"failed": [{"error_message": "bad audio codec"}]}
    fake_job = _FakeJobHandle("job-1")
    fake_job.get_file_results = _fake_get_file_results
    monkeypatch.setattr(sarvam_batch, "submit_job", lambda path: fake_job)

    try:
        pipeline_v3.DemoAuditPipelineV3._batch_stt(_pipeline_stub(), "session-1", [(wav_path, 0.0)])
        assert False, "expected SttJobFailedError"
    except sarvam_batch.SttJobFailedError as exc:
        assert "bad audio codec" in str(exc)

    row = sarvam_job_store.get("session-1")
    assert row["status"] == "stt_failed"
    assert "bad audio codec" in row["error_message"]


def test_batch_stt_multi_segment_processes_all_and_merges_sorted(tmp_path, monkeypatch):
    """N=4 here is illustrative only — the same code path must hold for any N
    (see test_sarvam_job_store.py's N=10/N=1000 test for the store layer)."""
    _fresh_job_store(tmp_path, monkeypatch)
    wav_paths = [tmp_path / f"seg{i}.wav" for i in range(4)]
    for p in wav_paths:
        p.write_bytes(b"x")

    def _fake_submit(path):
        return _FakeJobHandle(f"job-{path.stem}")

    monkeypatch.setattr(sarvam_batch, "submit_job", _fake_submit)
    monkeypatch.setattr(sarvam_batch, "upload_and_start", lambda job, path: None)
    monkeypatch.setattr(sarvam_batch, "poll_with_backoff",
                         lambda job, sid, on_status=None, is_cancelled=None: "Completed")

    def _fake_download(job, path, out_dir):
        # Each segment's single utterance start time encodes which segment
        # it is, so we can verify the final merge/sort is correct.
        idx = int(path.stem.replace("seg", ""))
        return sarvam_batch.BatchTranscriptResult(
            transcript="x", language_code="en-IN", has_diarization=True,
            segments=[sarvam_batch.DiarizedSegment(text=f"seg{idx}", start=0.0, end=1.0, speaker_id="0")],
        )
    monkeypatch.setattr(sarvam_batch, "download_and_parse", _fake_download)

    # Segments given with DEscending time offsets — final utterances must
    # still come back sorted by actual (offset-adjusted) start_time, proving
    # the merge doesn't just preserve processing/completion order.
    segments = [(wav_paths[i], float(3 - i) * 100) for i in range(4)]
    utterances = pipeline_v3.DemoAuditPipelineV3._batch_stt(_pipeline_stub(), "parent-1", segments)

    assert len(utterances) == 4
    starts = [u.start_time for u in utterances]
    assert starts == sorted(starts)

    progress = sarvam_job_store.segment_progress("parent-1")
    assert progress["total_segments"] == 4
    assert progress["completed"] == 4
    assert progress["failed"] == 0


def test_batch_stt_multi_segment_skips_sarvam_entirely_for_cached_segment(tmp_path, monkeypatch):
    _fresh_job_store(tmp_path, monkeypatch)
    wav_paths = [tmp_path / f"seg{i}.wav" for i in range(2)]
    for p in wav_paths:
        p.write_bytes(b"x")

    # Pre-populate segment 0 as already completed with a cached transcript —
    # simulates a resume after segment 0 succeeded on a prior (interrupted) run.
    sarvam_job_store.create_pending_segments("parent-1", [str(p) for p in wav_paths], now="t0")
    sarvam_job_store.upsert_submitted("parent-1::seg0", "old-job-0", str(wav_paths[0]), 100.0, 2, now="t0")
    sarvam_job_store.save_transcript(
        "parent-1::seg0",
        json.dumps({
            "transcript": "cached", "language_code": "en-IN", "has_diarization": True,
            "segments": [{"text": "cached text", "start": 0.0, "end": 1.0, "speaker_id": "0"}],
        }),
        now="t1",
    )

    calls = {"submit_job": 0, "resolve_existing_job_state": 0}
    monkeypatch.setattr(sarvam_batch, "submit_job",
                         lambda path: calls.__setitem__("submit_job", calls["submit_job"] + 1) or _FakeJobHandle("new-job"))
    monkeypatch.setattr(sarvam_batch, "resolve_existing_job_state",
                         lambda job_id: calls.__setitem__("resolve_existing_job_state", calls["resolve_existing_job_state"] + 1) or "Completed")
    monkeypatch.setattr(sarvam_batch, "upload_and_start", lambda job, path: None)
    monkeypatch.setattr(sarvam_batch, "poll_with_backoff",
                         lambda job, sid, on_status=None, is_cancelled=None: "Completed")
    monkeypatch.setattr(sarvam_batch, "download_and_parse", lambda job, path, out_dir: sarvam_batch.BatchTranscriptResult(
        transcript="fresh", language_code="en-IN", has_diarization=True,
        segments=[sarvam_batch.DiarizedSegment(text="fresh text", start=0.0, end=1.0, speaker_id="0")],
    ))

    utterances = pipeline_v3.DemoAuditPipelineV3._batch_stt(
        _pipeline_stub(), "parent-1", [(wav_paths[0], 0.0), (wav_paths[1], 1000.0)])

    texts = {u.native_text for u in utterances}
    assert "cached text" in texts     # segment 0 came from cache
    assert "fresh text" in texts      # segment 1 was actually processed
    # Segment 0's cached path never called submit_job or even a status check.
    assert calls["submit_job"] == 1          # only for segment 1
    assert calls["resolve_existing_job_state"] == 0   # cache short-circuit skips this check entirely


def test_batch_stt_multi_segment_one_failure_does_not_stop_others(tmp_path, monkeypatch):
    _fresh_job_store(tmp_path, monkeypatch)
    wav_paths = [tmp_path / f"seg{i}.wav" for i in range(3)]
    for p in wav_paths:
        p.write_bytes(b"x")

    monkeypatch.setattr(sarvam_batch, "submit_job", lambda path: _FakeJobHandle(f"job-{path.stem}"))
    monkeypatch.setattr(sarvam_batch, "upload_and_start", lambda job, path: None)

    def _fake_poll(job, sid, on_status=None, is_cancelled=None):
        return "Failed" if "seg1" in sid else "Completed"
    monkeypatch.setattr(sarvam_batch, "poll_with_backoff", _fake_poll)

    fail_job = _FakeJobHandle("job-seg1")
    fail_job.get_file_results = lambda: {"failed": [{"error_message": "bad audio"}]}
    monkeypatch.setattr(sarvam_batch, "submit_job",
                         lambda path: fail_job if "seg1" in path.stem else _FakeJobHandle(f"job-{path.stem}"))
    monkeypatch.setattr(sarvam_batch, "download_and_parse", lambda job, path, out_dir: sarvam_batch.BatchTranscriptResult(
        transcript="ok", language_code="en-IN", has_diarization=True,
        segments=[sarvam_batch.DiarizedSegment(text="ok", start=0.0, end=1.0, speaker_id="0")],
    ))

    try:
        pipeline_v3.DemoAuditPipelineV3._batch_stt(
            _pipeline_stub(), "parent-1", [(wav_paths[i], float(i) * 100) for i in range(3)])
        assert False, "expected SttJobFailedError"
    except sarvam_batch.SttJobFailedError as exc:
        assert "1/3 segment(s) failed" in str(exc)

    progress = sarvam_job_store.segment_progress("parent-1")
    assert progress["completed"] == 2   # segments 0 and 2 still succeeded
    assert progress["failed"] == 1      # only segment 1


def test_batch_stt_marks_cancelled_on_cooperative_cancel(tmp_path, monkeypatch):
    _fresh_job_store(tmp_path, monkeypatch)
    wav_path = tmp_path / "call.wav"
    wav_path.write_bytes(b"x")

    monkeypatch.setattr(sarvam_batch, "submit_job", lambda path: _FakeJobHandle("job-1"))
    monkeypatch.setattr(sarvam_batch, "upload_and_start", lambda job, path: None)
    monkeypatch.setattr(sarvam_batch, "poll_with_backoff",
                         lambda job, sid, on_status=None, is_cancelled=None: "Cancelled")

    try:
        pipeline_v3.DemoAuditPipelineV3._batch_stt(_pipeline_stub(), "session-1", [(wav_path, 0.0)])
        assert False, "expected SttJobFailedError (cancelled)"
    except sarvam_batch.SttJobFailedError:
        pass

    assert sarvam_job_store.get("session-1")["status"] == "cancelled"
