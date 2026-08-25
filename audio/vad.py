"""
audio/vad.py
Voice Activity Detection using Silero VAD (primary) with WebRTC fallback.

Silero VAD runs locally via ONNX — no GPU required, extremely fast.
Returns list of (start_sec, end_sec) speech segments.
"""
from __future__ import annotations
import logging
import struct
import wave
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)

SILERO_SAMPLE_RATE  = 16000
SILERO_THRESHOLD    = 0.50       # speech probability threshold
SILERO_MIN_SPEECH   = 0.25       # min speech segment duration (seconds)
SILERO_MIN_SILENCE  = 0.30       # silence gap that ends a speech segment


class SpeechSegment(NamedTuple):
    start: float   # seconds
    end:   float
    prob:  float   # avg Silero probability


class SileroVAD:
    """
    Thin wrapper around Silero VAD via silero-vad package or torch hub.
    Falls back to energy-based VAD if Silero is unavailable.
    """

    def __init__(self):
        self._model = None
        self._utils = None
        self._available = False
        self._try_load()

    def _try_load(self) -> None:
        try:
            import torch
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
            self._model = model
            self._utils = utils
            self._available = True
            logger.info("Silero VAD loaded via torch.hub")
        except Exception as exc:
            logger.warning(f"Silero VAD unavailable ({exc}); will use energy-based fallback")

    def detect(self, wav_path: Path) -> list[SpeechSegment]:
        """
        Run VAD on a 16kHz mono WAV file.
        Returns list of SpeechSegment(start, end, prob).
        """
        if self._available:
            try:
                return self._detect_silero(wav_path)
            except Exception as exc:
                logger.warning(f"Silero inference failed ({exc}); using energy fallback")
        return self._detect_energy(wav_path)

    def _detect_silero(self, wav_path: Path) -> list[SpeechSegment]:
        import torch

        get_speech_timestamps, _, read_audio, _, _ = self._utils

        wav = read_audio(str(wav_path), sampling_rate=SILERO_SAMPLE_RATE)
        raw_segments = get_speech_timestamps(
            wav,
            self._model,
            sampling_rate=SILERO_SAMPLE_RATE,
            threshold=SILERO_THRESHOLD,
            min_speech_duration_ms=int(SILERO_MIN_SPEECH * 1000),
            min_silence_duration_ms=int(SILERO_MIN_SILENCE * 1000),
            return_seconds=True,
        )

        result: list[SpeechSegment] = []
        for seg in raw_segments:
            result.append(SpeechSegment(
                start=float(seg["start"]),
                end=float(seg["end"]),
                prob=1.0,    # Silero doesn't return per-segment prob in this API
            ))
        logger.debug(f"Silero found {len(result)} speech segments")
        return result

    def _detect_energy(self, wav_path: Path) -> list[SpeechSegment]:
        """
        Simple energy-based fallback: 30ms frames, RMS threshold.
        Not as accurate as Silero but requires zero dependencies.
        """
        FRAME_MS  = 30
        THRESHOLD = 300    # RMS units (16-bit PCM)

        try:
            with wave.open(str(wav_path), "rb") as wf:
                sr        = wf.getframerate()
                sampwidth = wf.getsampwidth()
                n_frames  = wf.getnframes()
                raw       = wf.readframes(n_frames)
        except Exception as exc:
            logger.error(f"Cannot read WAV for energy VAD: {exc}")
            return []

        frame_size = int(sr * FRAME_MS / 1000)
        fmt = {1: "b", 2: "h", 4: "i"}.get(sampwidth, "h")
        total_samples = len(raw) // sampwidth
        samples = struct.unpack(f"<{total_samples}{fmt}", raw[:total_samples * sampwidth])

        segments: list[SpeechSegment] = []
        in_speech = False
        seg_start = 0.0
        i = 0

        while i < total_samples:
            chunk = samples[i: i + frame_size]
            if not chunk:
                break
            rms = (sum(s * s for s in chunk) / len(chunk)) ** 0.5
            t = i / sr
            if rms > THRESHOLD and not in_speech:
                in_speech = True
                seg_start = t
            elif rms <= THRESHOLD and in_speech:
                if t - seg_start >= SILERO_MIN_SPEECH:
                    segments.append(SpeechSegment(seg_start, t, rms / 32768))
                in_speech = False
            i += frame_size

        if in_speech:
            t = total_samples / sr
            if t - seg_start >= SILERO_MIN_SPEECH:
                segments.append(SpeechSegment(seg_start, t, 0.5))

        logger.debug(f"Energy VAD found {len(segments)} speech segments")
        return segments

    def speech_ratio(self, segments: list[SpeechSegment], total_duration: float) -> float:
        if total_duration <= 0:
            return 0.0
        speech_secs = sum(s.end - s.start for s in segments)
        return min(1.0, speech_secs / total_duration)


# Module-level singleton
_vad: SileroVAD | None = None


def get_vad() -> SileroVAD:
    global _vad
    if _vad is None:
        _vad = SileroVAD()
    return _vad
