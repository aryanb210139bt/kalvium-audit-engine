"""
transcription/sarvam_batch.py
Sarvam AI Batch Speech-to-Text — replaces the old "split into hundreds of
28-second chunks, POST each one to /speech-to-text" approach with a single
asynchronous Batch API job per recording (or per >2h split segment).

Grounded against the *actually installed* `sarvamai==0.1.31` SDK source
(verified 2026-09-02 by extracting the wheel and reading
sarvamai/speech_to_text_job/{job,client}.py, sarvamai/types/*.py directly —
not from memory, not from docs alone). Every method/field used below exists
in that source. Nothing here is a raw/manual HTTP call to a guessed
endpoint — the SDK owns the exact request/response shapes and the presigned
-URL upload headers (x-ms-blob-type etc.), which are otherwise undocumented.

Lifecycle (per requirement — initialise / upload / start / track / download):
  1. client.speech_to_text_job.create_job(...)   -> initialises the job,
     returns a SpeechToTextJob handle with .job_id already populated.
  2. job.upload_files([wav_path])                -> uploads the audio.
  3. job.start()                                  -> begins processing.
  4. job.get_status()                             -> job_state one of
     Accepted | Pending | Running | Completed | Failed. Polled with a
     capped exponential backoff (poll_with_backoff), never a tight loop.
  5. job.download_outputs(out_dir)                -> writes
     "<input_basename>.json" containing transcript / timestamps /
     diarized_transcript.

Model: saaras:v3, mode="transcribe" (native-language, normalised text —
matches what the rest of the pipeline already expects; English translation
still happens afterwards via the existing SpeechToText._translate_to_english,
unchanged). with_diarization=True always, since distinguishing Counsellor /
Student / Parent is required.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

SARVAM_BATCH_MODEL = "saaras:v3"
SARVAM_BATCH_MODE = "transcribe"

# Batch API hard limit is 2 hours/file; keep a safety margin so a slightly
# long recording (clock drift, container overhead) never gets rejected.
BATCH_MAX_SECONDS = 2 * 3600 - 5 * 60   # 1h55m

# Poll cadence: never hammer the API. Configurable via SARVAM_BATCH_POLL_*
# env vars (config/settings.py) — start modest, back off geometrically,
# cap so a long job still gets picked up promptly after finishing.


class SttJobFailedError(RuntimeError):
    """Raised when a Sarvam Batch STT job reaches a terminal Failed state,
    or cannot be parsed after completing. Caught specifically by the
    pipeline/API layer to mark the audit STT_FAILED (distinct from a
    generic pipeline failure) and to preserve the already-converted WAV so
    a retry can resubmit without redoing FFmpeg/download."""


@dataclass
class DiarizedSegment:
    text: str
    start: float
    end: float
    speaker_id: str


@dataclass
class BatchTranscriptResult:
    transcript: str
    language_code: str
    segments: list[DiarizedSegment] = field(default_factory=list)
    has_diarization: bool = False


def _client():
    """Constructs a fresh SarvamAI client per call — cheap (no network I/O
    at construction time) and avoids holding a long-lived client across the
    minutes-long polling loop."""
    from sarvamai import SarvamAI
    if not settings.sarvam_api_key:
        raise SttJobFailedError("SARVAM_API_KEY not configured")
    return SarvamAI(api_subscription_key=settings.sarvam_api_key)


# ── Step 1: initialise + upload + start ─────────────────────────────────────

def submit_job(wav_path: Path, num_speakers: Optional[int] = None, language_code: str = "unknown"):
    """
    Runs initialise -> upload -> start and returns the live SpeechToTextJob
    handle (job.job_id is populated the moment initialise() returns, i.e.
    before upload/start even run — callers should persist it immediately).
    """
    num_speakers = num_speakers if num_speakers is not None else settings.sarvam_batch_num_speakers
    client = _client()
    job = client.speech_to_text_job.create_job(
        model=SARVAM_BATCH_MODEL,
        mode=SARVAM_BATCH_MODE,
        with_diarization=True,
        with_timestamps=True,
        language_code=language_code,
        num_speakers=num_speakers,
    )
    logger.info(f"[sarvam-batch] job {job.job_id}: initialised (model={SARVAM_BATCH_MODEL}, "
                f"diarization=on, num_speakers={num_speakers})")
    return job


def upload_and_start(job, wav_path: Path) -> None:
    """Runs steps 2 & 3 (upload, start) against an already-initialised job
    handle. Kept as two explicit SDK calls (not the higher-level
    convenience wrapper) so each step's own exception is distinguishable
    in logs/status updates."""
    job.upload_files([str(wav_path)])
    logger.info(f"[sarvam-batch] job {job.job_id}: upload complete ({wav_path.name})")
    job.start()
    logger.info(f"[sarvam-batch] job {job.job_id}: started")


def get_job_handle(job_id: str):
    """Reconstructs a job handle from a persisted job_id alone — this is
    what makes the lifecycle survive a page refresh or server restart: no
    in-memory state is needed, only the job_id from sarvam_job_store."""
    return _client().speech_to_text_job.get_job(job_id)


# ── Step 2: poll with capped exponential backoff ────────────────────────────

def poll_with_backoff(job, session_id: str, on_status=None, is_cancelled=None) -> str:
    """
    Polls job.get_status() with a capped exponential backoff (never a tight
    loop) — cadence configurable via SARVAM_BATCH_POLL_* settings. Returns
    the final job_state ("Completed"/"Failed"), or "Cancelled" if
    `is_cancelled()` returns True between polls (cooperative cancellation —
    we can't kill Sarvam's own job, only stop waiting on/paying attention
    to it; the already-submitted job keeps running on Sarvam's side and a
    later Resume will pick up its result rather than resubmitting).

    `on_status(status)` is called after every poll for logging / DB
    persistence — see pipeline_v3._batch_stt.
    """
    interval = settings.sarvam_batch_poll_initial_sec
    poll_max = settings.sarvam_batch_poll_max_sec
    backoff = settings.sarvam_batch_poll_backoff
    timeout = settings.sarvam_batch_poll_timeout_sec

    started = time.monotonic()
    polls = 0
    while True:
        if is_cancelled and is_cancelled():
            logger.info(f"[sarvam-batch] job {job.job_id} (session {session_id}): "
                        f"cancellation requested — stopping poll (job keeps running on Sarvam's side)")
            return "Cancelled"

        status = job.get_status()
        state = status.job_state
        polls += 1
        elapsed = time.monotonic() - started
        logger.info(f"[sarvam-batch] job {job.job_id} (session {session_id}): "
                    f"poll #{polls} -> {state} ({elapsed:.0f}s elapsed)")
        if on_status:
            on_status(status)
        if state in ("Completed", "Failed"):
            return state
        if elapsed > timeout:
            raise SttJobFailedError(
                f"Sarvam job {job.job_id} did not finish within {timeout}s (last state: {state})"
            )
        time.sleep(interval)
        interval = min(interval * backoff, poll_max)


# ── Step 3: download + parse ─────────────────────────────────────────────────

def download_and_parse(job, wav_path: Path, out_dir: Path) -> BatchTranscriptResult:
    """
    Downloads the job's output (job.download_outputs writes
    "<wav_path.name>.json" into out_dir) and parses it into a
    BatchTranscriptResult. Raises SttJobFailedError if the file/fields are
    missing rather than silently returning an empty transcript.
    """
    import json as _json

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        job.download_outputs(str(out_dir))
    except Exception as exc:
        raise SttJobFailedError(f"download_outputs failed for job {job.job_id}: {exc}") from exc

    result_path = out_dir / f"{wav_path.name}.json"
    if not result_path.exists():
        raise SttJobFailedError(
            f"Sarvam job {job.job_id} completed but no output file found at {result_path}"
        )

    try:
        raw = _json.loads(result_path.read_text())
    except Exception as exc:
        raise SttJobFailedError(f"Could not parse Sarvam output JSON for job {job.job_id}: {exc}") from exc

    transcript = (raw.get("transcript") or "").strip()
    language_code = raw.get("language_code") or "unknown"

    segments: list[DiarizedSegment] = []
    diarized = raw.get("diarized_transcript") or {}
    entries = diarized.get("entries") or []
    for e in entries:
        text = (e.get("transcript") or "").strip()
        if not text:
            continue
        segments.append(DiarizedSegment(
            text=text,
            start=float(e.get("start_time_seconds", 0.0)),
            end=float(e.get("end_time_seconds", 0.0)),
            speaker_id=str(e.get("speaker_id", "0")),
        ))

    has_diarization = len(segments) > 0
    if not has_diarization:
        # Diarization is documented as beta and may be omitted even when
        # requested. Fall back to the plain chunk-level `timestamps` field
        # so we still preserve real timestamps (requirement: "preserve
        # timestamps and speaker labels") — speaker role assignment for
        # this fallback case is handled by diarization.diarizer's existing
        # acoustic methods (stereo/mono-energy) in pipeline_v3._batch_stt.
        ts = raw.get("timestamps") or {}
        words = ts.get("words") or []
        starts = ts.get("start_time_seconds") or []
        ends = ts.get("end_time_seconds") or []
        for i, chunk_text in enumerate(words):
            if not chunk_text.strip():
                continue
            segments.append(DiarizedSegment(
                text=chunk_text.strip(),
                start=float(starts[i]) if i < len(starts) else 0.0,
                end=float(ends[i]) if i < len(ends) else 0.0,
                speaker_id="UNKNOWN",
            ))
        logger.warning(f"[sarvam-batch] job {job.job_id}: no diarized_transcript in output "
                        f"(beta feature) — falling back to {len(segments)} timestamped chunks, "
                        f"speaker roles will use acoustic diarization")

    logger.info(f"[sarvam-batch] job {job.job_id}: parsed {len(segments)} segment(s), "
                f"language={language_code}, diarized={has_diarization}")

    return BatchTranscriptResult(
        transcript=transcript,
        language_code=language_code,
        segments=segments,
        has_diarization=has_diarization,
    )


# ── Restart / retry support ──────────────────────────────────────────────────

def resolve_existing_job_state(job_id: str) -> Optional[str]:
    """
    Live-checks a previously-submitted job_id's current state on Sarvam's
    side. Used on retry so we never blindly resubmit (and re-pay for) a
    job that actually already completed while our local poller was
    interrupted (server restart) — see sarvam_job_store.recover_orphaned.
    Returns None if the job can't be found/queried at all (safe to treat
    as "submit a fresh one").
    """
    try:
        job = get_job_handle(job_id)
        return job.get_status().job_state
    except Exception as exc:
        logger.warning(f"[sarvam-batch] could not resolve existing job {job_id}: {exc}")
        return None
