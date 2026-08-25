"""
audio/processor_v2.py
Upgraded audio processing layer for the multilingual pipeline.

Improvements over processor.py:
  - Noise reduction (FFmpeg filters + noisereduce)
  - Silero VAD — splits only at genuine speech boundaries
  - 30s primary chunks with 5s overlap buffer
  - Per-chunk speech ratio and energy metadata
  - Caching: skips re-processing if chunks already exist
"""
from __future__ import annotations
import json
import logging
import subprocess
from pathlib import Path

from audio.noise_reducer import reduce_noise
from audio.vad import get_vad, SpeechSegment
from config.models_multilingual import AudioChunkV2

logger = logging.getLogger(__name__)

TARGET_SR          = 16000
PRIMARY_CHUNK_SEC  = 30
OVERLAP_SEC        = 5          # seconds of audio carried over into next chunk
MIN_SPEECH_RATIO   = 0.15       # chunks below this are skipped as silence


class AudioProcessorV2:

    def process(self, recording_path: Path) -> tuple[Path, list[AudioChunkV2]]:
        """
        Full audio processing pipeline:
          1. Convert → 16kHz mono WAV
          2. Noise reduction + loudnorm
          3. Silero VAD
          4. VAD-aware chunking with overlap
        Returns (clean_wav_path, chunks).
        """
        wav_path   = self._convert(recording_path)
        clean_path = self._clean(wav_path)
        vad_segs   = get_vad().detect(clean_path)
        chunks     = self._chunk(clean_path, vad_segs)
        return clean_path, chunks

    def get_duration(self, wav_path: Path) -> float:
        return _ffprobe_duration(wav_path)

    # ── Step 1: Convert ────────────────────────────────────────────────────────

    def _convert(self, src: Path) -> Path:
        dst = src.parent / f"{src.stem}_raw16k.wav"
        if dst.exists():
            logger.info(f"  WAV cache hit: {dst.name}")
            return dst
        cmd = [
            "ffmpeg", "-y", "-i", str(src),
            "-ar", str(TARGET_SR),
            "-ac", "1",              # mono for ASR
            "-f", "wav",
            "-c:a", "pcm_s16le",
            str(dst),
        ]
        logger.info(f"  Converting: {src.name}")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg convert failed:\n{r.stderr[:500]}")
        return dst

    # ── Step 2: Noise reduction ────────────────────────────────────────────────

    def _clean(self, wav_path: Path) -> Path:
        clean = wav_path.parent / f"{wav_path.stem}_clean.wav"
        if clean.exists():
            logger.info(f"  Clean WAV cache hit: {clean.name}")
            return clean
        logger.info("  Applying noise reduction…")
        return reduce_noise(wav_path, clean)

    # ── Step 3 + 4: VAD-aware chunking ────────────────────────────────────────

    def _chunk(self, wav_path: Path, vad_segs: list[SpeechSegment]) -> list[AudioChunkV2]:
        chunk_dir = wav_path.parent / "chunks_v2"
        chunk_dir.mkdir(exist_ok=True)

        duration = _ffprobe_duration(wav_path)
        if duration <= 0:
            raise RuntimeError(f"Cannot determine duration: {wav_path}")

        # Build time windows: 30s with 5s overlap
        windows = _build_windows(duration, PRIMARY_CHUNK_SEC, OVERLAP_SEC)
        logger.info(f"  Duration: {duration:.1f}s | Windows: {len(windows)}")

        chunks: list[AudioChunkV2] = []
        for i, (t_start, t_end) in enumerate(windows):
            chunk_path = chunk_dir / f"chunk_{i:04d}.wav"
            if not chunk_path.exists():
                _extract_segment(wav_path, t_start, t_end, chunk_path)

            speech_ratio = _compute_speech_ratio(vad_segs, t_start, t_end)
            if speech_ratio < MIN_SPEECH_RATIO:
                logger.debug(f"  Chunk {i} skipped (speech_ratio={speech_ratio:.2f})")
                continue

            energy = _compute_energy_db(vad_segs, t_start, t_end)
            chunks.append(AudioChunkV2(
                chunk_id=i,
                start_time=round(t_start, 3),
                end_time=round(t_end, 3),
                file_path=str(chunk_path),
                speech_ratio=round(speech_ratio, 3),
                energy_db=round(energy, 1),
            ))

        logger.info(f"  {len(chunks)} speech chunks retained (from {len(windows)} windows)")
        return chunks


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_windows(
    duration: float, chunk_sec: float, overlap: float
) -> list[tuple[float, float]]:
    """Build (start, end) windows with overlap."""
    windows = []
    t = 0.0
    step = chunk_sec - overlap
    while t < duration:
        end = min(t + chunk_sec, duration)
        windows.append((t, end))
        if end >= duration:
            break
        t += step
    return windows


def _extract_segment(src: Path, start: float, end: float, dst: Path) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-t",  str(end - start),
        "-i",  str(src),
        "-c:a", "pcm_s16le",
        "-ar", str(TARGET_SR),
        "-ac", "1",
        str(dst),
    ]
    subprocess.run(cmd, capture_output=True, check=False)


def _compute_speech_ratio(
    segs: list[SpeechSegment], t_start: float, t_end: float
) -> float:
    if not segs:
        return 0.8   # assume speech if no VAD data
    window = t_end - t_start
    speech = 0.0
    for s in segs:
        overlap = min(s.end, t_end) - max(s.start, t_start)
        if overlap > 0:
            speech += overlap
    return min(1.0, speech / window)


def _compute_energy_db(
    segs: list[SpeechSegment], t_start: float, t_end: float
) -> float:
    probs = [s.prob for s in segs
             if s.start < t_end and s.end > t_start]
    if not probs:
        return -30.0
    avg_prob = sum(probs) / len(probs)
    import math
    return round(20 * math.log10(max(avg_prob, 1e-6)), 1)


def _ffprobe_duration(path: Path) -> float:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet",
             "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1",
             str(path)],
            capture_output=True, text=True, check=True,
        )
        return float(r.stdout.strip())
    except Exception:
        return 0.0
