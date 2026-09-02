"""
tests/test_sarvam_batch.py
Unit tests for transcription/sarvam_batch.py — the Sarvam Batch STT
lifecycle (initialise/upload/start/poll/download). Everything here is
mocked (no real network call, no real Sarvam API key needed) so these run
fully offline and fast. Live end-to-end verification against the real
Sarvam API is done separately (see the manual verification script), not
in this automated suite.
"""
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))

import transcription.sarvam_batch as sb


# ── Fakes standing in for the real sarvamai SDK objects ─────────────────────

class FakeStatus:
    def __init__(self, job_id, job_state, total_files=1, ok=0, failed=0):
        self.job_id = job_id
        self.job_state = job_state
        self.total_files = total_files
        self.successful_files_count = ok
        self.failed_files_count = failed


class FakeJob:
    """Stands in for sarvamai's SpeechToTextJob handle."""

    def __init__(self, job_id, states=("Completed",), output_payload=None):
        self.job_id = job_id
        self._states = list(states)
        self._poll_index = 0
        self.uploaded_files = None
        self.started = False
        self._output_payload = output_payload or {}

    def upload_files(self, paths):
        self.uploaded_files = list(paths)

    def start(self):
        self.started = True

    def get_status(self):
        state = self._states[min(self._poll_index, len(self._states) - 1)]
        self._poll_index += 1
        return FakeStatus(self.job_id, state)

    def get_file_results(self):
        return {"successful": [], "failed": [{"error_message": "synthetic failure"}]}

    def download_outputs(self, out_dir):
        out_path = Path(out_dir) / f"{self._wav_name}.json"
        out_path.write_text(json.dumps(self._output_payload))

    def set_wav_name_for_download(self, name):
        self._wav_name = name


class FakeSTTJobClient:
    def __init__(self, job_to_return=None):
        self.created_kwargs = []
        self._job_to_return = job_to_return

    def create_job(self, **kwargs):
        self.created_kwargs.append(kwargs)
        return self._job_to_return or FakeJob("job-created")

    def get_job(self, job_id):
        return self._job_to_return or FakeJob(job_id)


class FakeSarvamClient:
    def __init__(self, job_to_return=None):
        self.speech_to_text_job = FakeSTTJobClient(job_to_return)


# ── submit_job ────────────────────────────────────────────────────────────

def test_submit_job_uses_saaras_v3_with_diarization_and_timestamps(monkeypatch, tmp_path):
    fake_job = FakeJob("job-xyz")
    fake_client = FakeSarvamClient(job_to_return=fake_job)
    monkeypatch.setattr(sb, "_client", lambda: fake_client)

    job = sb.submit_job(tmp_path / "call.wav")

    assert job.job_id == "job-xyz"
    kwargs = fake_client.speech_to_text_job.created_kwargs[0]
    assert kwargs["model"] == "saaras:v3"
    assert kwargs["mode"] == "transcribe"
    assert kwargs["with_diarization"] is True
    assert kwargs["with_timestamps"] is True
    assert kwargs["num_speakers"] == sb.settings.sarvam_batch_num_speakers


def test_submit_job_num_speakers_override(monkeypatch, tmp_path):
    fake_client = FakeSarvamClient(job_to_return=FakeJob("job-1"))
    monkeypatch.setattr(sb, "_client", lambda: fake_client)
    sb.submit_job(tmp_path / "call.wav", num_speakers=2)
    assert fake_client.speech_to_text_job.created_kwargs[0]["num_speakers"] == 2


def test_client_raises_without_api_key(monkeypatch):
    monkeypatch.setattr(sb.settings, "sarvam_api_key", "")
    try:
        sb._client()
        assert False, "expected SttJobFailedError"
    except sb.SttJobFailedError:
        pass


# ── upload_and_start ──────────────────────────────────────────────────────

def test_upload_and_start_calls_upload_then_start(tmp_path):
    job = FakeJob("job-1")
    wav = tmp_path / "call.wav"
    sb.upload_and_start(job, wav)
    assert job.uploaded_files == [str(wav)]
    assert job.started is True


# ── poll_with_backoff ─────────────────────────────────────────────────────

def test_poll_with_backoff_returns_completed_without_sleeping_forever(monkeypatch):
    job = FakeJob("job-1", states=["Pending", "Running", "Completed"])
    sleeps = []
    monkeypatch.setattr(sb.time, "sleep", lambda s: sleeps.append(s))

    state = sb.poll_with_backoff(job, "session-1")

    assert state == "Completed"
    assert job._poll_index == 3          # exactly 3 polls, no extra
    assert len(sleeps) == 2               # slept between poll 1->2 and 2->3, not after the last
    assert sleeps[0] == sb.settings.sarvam_batch_poll_initial_sec


def test_poll_with_backoff_backs_off_geometrically_and_caps(monkeypatch):
    # Enough Pending states to exceed the cap at least once.
    job = FakeJob("job-1", states=["Pending"] * 6 + ["Completed"])
    sleeps = []
    monkeypatch.setattr(sb.time, "sleep", lambda s: sleeps.append(s))

    sb.poll_with_backoff(job, "session-1")

    for s in sleeps:
        assert s <= sb.settings.sarvam_batch_poll_max_sec
    # Strictly increasing until it hits the cap.
    assert sleeps == sorted(sleeps)


def test_poll_with_backoff_raises_on_timeout(monkeypatch):
    job = FakeJob("job-1", states=["Running"] * 100)   # never completes
    monkeypatch.setattr(sb.time, "sleep", lambda s: None)
    monkeypatch.setattr(sb.settings, "sarvam_batch_poll_timeout_sec", 0)  # instantly "expired"

    # Fake monotonic clock: first call = 0 (start), every call after = 1 (elapsed > 0 = timeout)
    clock = iter([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    monkeypatch.setattr(sb.time, "monotonic", lambda: next(clock, 999))

    try:
        sb.poll_with_backoff(job, "session-1")
        assert False, "expected SttJobFailedError on timeout"
    except sb.SttJobFailedError:
        pass


def test_poll_with_backoff_stops_on_cancellation_without_extra_polls(monkeypatch):
    job = FakeJob("job-1", states=["Running"] * 10)
    monkeypatch.setattr(sb.time, "sleep", lambda s: None)

    state = sb.poll_with_backoff(job, "session-1", is_cancelled=lambda: True)

    assert state == "Cancelled"
    assert job._poll_index == 0   # never even called get_status() once


def test_poll_with_backoff_calls_on_status_every_poll(monkeypatch):
    job = FakeJob("job-1", states=["Pending", "Completed"])
    monkeypatch.setattr(sb.time, "sleep", lambda s: None)
    seen = []
    sb.poll_with_backoff(job, "session-1", on_status=lambda status: seen.append(status.job_state))
    assert seen == ["Pending", "Completed"]


# ── download_and_parse ────────────────────────────────────────────────────

def test_download_and_parse_extracts_diarized_segments(tmp_path):
    wav_path = tmp_path / "call.wav"
    payload = {
        "transcript": "hello world",
        "language_code": "hi-IN",
        "diarized_transcript": {
            "entries": [
                {"transcript": "namaste", "start_time_seconds": 0.0, "end_time_seconds": 2.0, "speaker_id": "0"},
                {"transcript": "kaise ho", "start_time_seconds": 2.0, "end_time_seconds": 4.5, "speaker_id": "1"},
            ]
        },
    }
    job = FakeJob("job-1", output_payload=payload)
    job.set_wav_name_for_download(wav_path.name)

    out_dir = tmp_path / "out"
    result = sb.download_and_parse(job, wav_path, out_dir)

    assert result.has_diarization is True
    assert result.language_code == "hi-IN"
    assert len(result.segments) == 2
    assert result.segments[0].speaker_id == "0"
    assert result.segments[0].text == "namaste"
    assert result.segments[1].end == 4.5


def test_download_and_parse_falls_back_to_timestamps_when_no_diarization(tmp_path):
    wav_path = tmp_path / "call.wav"
    payload = {
        "transcript": "hello world",
        "language_code": "en-IN",
        "timestamps": {
            "words": ["hello", "world"],
            "start_time_seconds": [0.0, 1.0],
            "end_time_seconds": [1.0, 2.0],
        },
        # no diarized_transcript key at all — simulates the documented beta
        # feature omitting it even when requested
    }
    job = FakeJob("job-1", output_payload=payload)
    job.set_wav_name_for_download(wav_path.name)

    result = sb.download_and_parse(job, wav_path, tmp_path / "out")

    assert result.has_diarization is False
    assert len(result.segments) == 2
    assert all(s.speaker_id == "UNKNOWN" for s in result.segments)


def test_download_and_parse_raises_if_output_file_missing(tmp_path):
    wav_path = tmp_path / "call.wav"

    class SilentJob(FakeJob):
        def download_outputs(self, out_dir):
            pass  # never writes the expected file

    job = SilentJob("job-1")
    try:
        sb.download_and_parse(job, wav_path, tmp_path / "out")
        assert False, "expected SttJobFailedError"
    except sb.SttJobFailedError:
        pass


# ── resolve_existing_job_state ───────────────────────────────────────────

def test_resolve_existing_job_state_returns_live_state(monkeypatch):
    fake_client = FakeSarvamClient(job_to_return=FakeJob("job-1", states=["Completed"]))
    monkeypatch.setattr(sb, "_client", lambda: fake_client)
    assert sb.resolve_existing_job_state("job-1") == "Completed"


def test_resolve_existing_job_state_returns_none_on_error(monkeypatch):
    def _boom():
        raise RuntimeError("network down")
    monkeypatch.setattr(sb, "_client", _boom)
    assert sb.resolve_existing_job_state("job-1") is None
