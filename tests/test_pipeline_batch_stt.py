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
import sys
import threading
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
    return SimpleNamespace(p=None, processor=AudioProcessor(), stt=_FakeSTT(), diarizer=Diarizer())


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
