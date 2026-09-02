"""
diarization/diarizer.py
Assign Counsellor / Student / Parent speaker labels to audio chunks.

Strategy (best to worst):
  1. Stereo file  → L-channel = Counsellor (outbound), R-channel = Student (inbound)
     Perfect for TeleCMI recordings. No config needed.
  2. Mono file    → Energy-based turn detection using 200ms analysis frames.
  3. pyannote     → Full ML diarization (needs HF_TOKEN).
  4. Mock         → Fallback pattern.

Pure stdlib + ffmpeg — no pydub required.
"""
from __future__ import annotations
import logging
import subprocess
import tempfile
from pathlib import Path

from audio.processor import read_wav_rms, extract_channel_wav
from config.models import AudioChunk, Speaker
from config.settings import get_settings

logger   = logging.getLogger(__name__)
settings = get_settings()

# L must be ≥20% louder than R for stereo assignment
STEREO_MIN_RATIO  = 1.20
# Silence gap (seconds) that indicates a speaker turn change in mono
TURN_SILENCE_SEC  = 0.5
# Analysis frame size for mono energy detection
FRAME_SEC         = 0.2


def assign_roles_by_talktime(durations: dict[str, float]) -> dict[str, Speaker]:
    """
    Map anonymous speaker labels (pyannote's "SPEAKER_00"/"SPEAKER_01"-style
    strings, or Sarvam Batch STT's numeric diarized_transcript speaker_id
    strings "0"/"1"/"2") to Counsellor/Student/Parent roles.

    This is the app's existing speaker-identification heuristic (previously
    inline in _diarize_pyannote only): the speaker who talks the most in a
    KNET-style demo/sales call is overwhelmingly the counsellor running the
    session, the second-most is the student, and a third participant (when
    present) is the parent. Anything beyond 3 distinct speakers is UNKNOWN
    rather than guessed. Reused as-is (not reimplemented) for Sarvam's Batch
    API diarization output — see transcription/sarvam_batch.py.
    """
    ranked = sorted(durations, key=durations.__getitem__, reverse=True)
    roles  = [Speaker.COUNSELLOR, Speaker.STUDENT, Speaker.PARENT]
    return {spk: roles[i] if i < len(roles) else Speaker.UNKNOWN
            for i, spk in enumerate(ranked)}


class Diarizer:

    def diarize(self, wav_path: Path, chunks: list[AudioChunk]) -> dict[int, Speaker]:
        if settings.diarization_provider == "pyannote":
            return self._diarize_pyannote(wav_path, chunks)
        if settings.diarization_provider == "mock":
            # Deliberately requested — skip real analysis entirely (fast dev/testing path).
            logger.info("DIARIZATION_PROVIDER=mock — using fallback pattern, no real analysis")
            return self._diarize_mock(chunks)

        try:
            n_channels = self._get_channels(wav_path)
            if n_channels >= 2:
                logger.info("Stereo detected → channel-based diarization (L=Counsellor, R=Student)")
                return self._diarize_stereo(wav_path, chunks)
            else:
                logger.info("Mono → energy turn-detection diarization")
                return self._diarize_mono_energy(wav_path, chunks)
        except Exception as exc:
            logger.warning(f"Energy diarization failed ({exc}) — using mock")
            return self._diarize_mock(chunks)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _get_channels(wav_path: Path) -> int:
        import wave as _wave
        try:
            with _wave.open(str(wav_path), "rb") as wf:
                return wf.getnchannels()
        except Exception:
            return 1

    # ── 1. Stereo channel diarization ─────────────────────────────────────────

    def _diarize_stereo(self, wav_path: Path, chunks: list[AudioChunk]) -> dict[int, Speaker]:
        """
        Extract L and R channels to temp files, compare per-chunk RMS.
        Higher RMS = dominant speaker for that chunk.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            left_wav  = Path(tmpdir) / "left.wav"
            right_wav = Path(tmpdir) / "right.wav"
            extract_channel_wav(wav_path, 0, left_wav)
            extract_channel_wav(wav_path, 1, right_wav)

            result: dict[int, Speaker] = {}
            c_count = 0
            for chunk in chunks:
                l_rms = read_wav_rms(left_wav,  chunk.start_time, chunk.end_time)
                r_rms = read_wav_rms(right_wav, chunk.start_time, chunk.end_time)

                if l_rms == 0 and r_rms == 0:
                    result[chunk.chunk_id] = Speaker.UNKNOWN
                elif r_rms == 0 or (l_rms > 0 and l_rms / max(r_rms, 0.001) >= STEREO_MIN_RATIO):
                    result[chunk.chunk_id] = Speaker.COUNSELLOR
                    c_count += 1
                elif l_rms == 0 or (r_rms > 0 and r_rms / max(l_rms, 0.001) >= STEREO_MIN_RATIO):
                    result[chunk.chunk_id] = Speaker.STUDENT
                else:
                    # Nearly equal — give slight preference to counsellor
                    result[chunk.chunk_id] = Speaker.COUNSELLOR if l_rms >= r_rms else Speaker.STUDENT
                    c_count += (1 if l_rms >= r_rms else 0)

        logger.info(f"Stereo diarization: {c_count}/{len(chunks)} → Counsellor")
        return result

    # ── 2. Mono energy turn detection ─────────────────────────────────────────

    def _diarize_mono_energy(self, wav_path: Path, chunks: list[AudioChunk]) -> dict[int, Speaker]:
        """
        Analyse mono audio in FRAME_SEC windows to find speech bursts.
        Assign first burst = Counsellor, alternate on silence gaps ≥ TURN_SILENCE_SEC.
        Map each chunk to the burst covering its midpoint.
        """
        duration = max((c.end_time for c in chunks), default=0)
        if duration <= 0:
            return self._diarize_mock(chunks)

        # Compute RMS for each frame
        n_frames = int(duration / FRAME_SEC) + 1
        energies = [
            read_wav_rms(wav_path, i * FRAME_SEC, (i + 1) * FRAME_SEC)
            for i in range(n_frames)
        ]

        # Silence threshold: 15% of mean non-zero energy
        nonzero = [e for e in energies if e > 0]
        if not nonzero:
            return self._diarize_mock(chunks)
        threshold = sum(nonzero) / len(nonzero) * 0.15

        # Group into speech bursts
        bursts: list[tuple[float, float, Speaker]] = []
        in_burst       = False
        burst_start    = 0.0
        silence_frames = 0
        cur_speaker    = Speaker.COUNSELLOR

        for i, energy in enumerate(energies):
            t = i * FRAME_SEC
            if energy >= threshold:
                if not in_burst:
                    burst_start = t
                    in_burst    = True
                silence_frames = 0
            else:
                if in_burst:
                    silence_frames += 1
                    if silence_frames * FRAME_SEC >= TURN_SILENCE_SEC:
                        burst_end = t - (silence_frames - 1) * FRAME_SEC
                        bursts.append((burst_start, burst_end, cur_speaker))
                        in_burst       = False
                        silence_frames = 0
                        cur_speaker    = (Speaker.STUDENT
                                          if cur_speaker == Speaker.COUNSELLOR
                                          else Speaker.COUNSELLOR)

        if in_burst:
            bursts.append((burst_start, duration, cur_speaker))

        if not bursts:
            return self._diarize_mock(chunks)

        def _assign(chunk: AudioChunk) -> Speaker:
            best_spk     = Speaker.UNKNOWN
            best_overlap = -1.0
            for b_start, b_end, spk in bursts:
                overlap = min(chunk.end_time, b_end) - max(chunk.start_time, b_start)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_spk     = spk
            return best_spk

        result = {c.chunk_id: _assign(c) for c in chunks}
        c_count = sum(1 for s in result.values() if s == Speaker.COUNSELLOR)
        s_count = sum(1 for s in result.values() if s == Speaker.STUDENT)
        logger.info(
            f"Mono energy diarization: {len(bursts)} turns → "
            f"{c_count} Counsellor / {s_count} Student"
        )
        return result

    # ── 3. pyannote ───────────────────────────────────────────────────────────

    def _diarize_pyannote(self, wav_path: Path, chunks: list[AudioChunk]) -> dict[int, Speaker]:
        try:
            from pyannote.audio import Pipeline
        except ImportError:
            logger.warning("pyannote not installed — using energy diarizer")
            return self.diarize(wav_path, chunks)

        if not settings.hf_token:
            logger.warning("HF_TOKEN not set — using energy diarizer")
            return self.diarize(wav_path, chunks)

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=settings.hf_token,
        )
        dia = pipeline(str(wav_path))

        durations: dict[str, float] = {}
        for turn, _, spk in dia.itertracks(yield_label=True):
            durations[spk] = durations.get(spk, 0) + turn.duration
        role_map = assign_roles_by_talktime(durations)

        result: dict[int, Speaker] = {}
        for chunk in chunks:
            overlaps: dict[str, float] = {}
            for turn, _, spk in dia.itertracks(yield_label=True):
                ov = min(turn.end, chunk.end_time) - max(turn.start, chunk.start_time)
                if ov > 0:
                    overlaps[spk] = overlaps.get(spk, 0) + ov
            dom = max(overlaps, key=overlaps.__getitem__) if overlaps else None
            result[chunk.chunk_id] = role_map.get(dom, Speaker.UNKNOWN) if dom else Speaker.UNKNOWN

        return result

    # ── 4. Mock fallback ──────────────────────────────────────────────────────

    @staticmethod
    def _diarize_mock(chunks: list[AudioChunk]) -> dict[int, Speaker]:
        pattern = [
            Speaker.COUNSELLOR, Speaker.COUNSELLOR, Speaker.STUDENT,
            Speaker.COUNSELLOR, Speaker.COUNSELLOR, Speaker.STUDENT,
        ]
        result = {
            c.chunk_id: (Speaker.COUNSELLOR if c.chunk_id == 0
                         else pattern[c.chunk_id % len(pattern)])
            for c in chunks
        }
        logger.info("Mock diarization applied (fallback)")
        return result
