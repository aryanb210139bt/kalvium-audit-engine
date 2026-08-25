"""
audio/noise_reducer.py
Noise reduction pipeline for call audio.

Stack (in order of application):
  1. High-pass filter (FFmpeg) — removes sub-80Hz rumble and handling noise
  2. LUFS normalization (FFmpeg loudnorm) — consistent volume across calls
  3. noisereduce (scipy) — spectral gating for stationary noise (AC, fan hum)

Each step is optional — if a library is unavailable, that step is skipped
and a warning is logged. FFmpeg steps always run (FFmpeg is required).
"""
from __future__ import annotations
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

TARGET_LUFS  = -23.0
TARGET_TP    = -1.0
TARGET_LRA   = 11.0


def reduce_noise(input_wav: Path, output_wav: Path) -> Path:
    """
    Apply the full noise reduction chain to a 16kHz mono WAV.
    Returns output_wav path (always written, even if some steps skip).
    """
    # Step 1 + 2: High-pass + loudnorm via FFmpeg (single pass for speed)
    ffmpeg_out = output_wav.parent / f"{output_wav.stem}_ffmpeg.wav"
    _ffmpeg_filter(input_wav, ffmpeg_out)

    # Step 3: Spectral gating via noisereduce
    spectral_out = output_wav
    success = _spectral_gate(ffmpeg_out, spectral_out)

    if not success:
        # noisereduce not available — rename FFmpeg output to final
        ffmpeg_out.rename(output_wav)
    else:
        ffmpeg_out.unlink(missing_ok=True)

    logger.info(f"Noise reduction complete → {output_wav.name}")
    return output_wav


def _ffmpeg_filter(src: Path, dst: Path) -> None:
    """High-pass + LUFS normalization in one FFmpeg pass."""
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-af", (
            f"highpass=f=80,"
            f"lowpass=f=8000,"
            f"loudnorm=I={TARGET_LUFS}:TP={TARGET_TP}:LRA={TARGET_LRA}"
        ),
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning(f"FFmpeg filter warning: {result.stderr[-300:]}")
        # Copy original as fallback
        import shutil
        shutil.copy2(src, dst)


def _spectral_gate(src: Path, dst: Path) -> bool:
    """
    Apply spectral gating using noisereduce.
    Returns True if successful, False if library not available.
    """
    try:
        import numpy as np
        import noisereduce as nr
        import wave
        import struct

        with wave.open(str(src), "rb") as wf:
            sr = wf.getframerate()
            sw = wf.getsampwidth()
            nf = wf.getnframes()
            raw = wf.readframes(nf)

        fmt = {1: "b", 2: "h", 4: "i"}.get(sw, "h")
        samples = np.array(struct.unpack(f"<{nf}{fmt}", raw[:nf * sw]), dtype=np.float32)
        samples /= 32768.0   # normalize to [-1, 1]

        # Use first 0.5s as noise profile if audio is long enough
        noise_clip = samples[:min(int(sr * 0.5), len(samples) // 10)]
        reduced = nr.reduce_noise(
            y=samples,
            sr=sr,
            y_noise=noise_clip,
            stationary=True,
            prop_decrease=0.75,
        )

        # Write back as 16-bit PCM WAV
        out_samples = (reduced * 32767).astype(np.int16)
        with wave.open(str(dst), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(out_samples.tobytes())

        return True

    except ImportError:
        logger.debug("noisereduce not installed — skipping spectral gate")
        return False
    except Exception as exc:
        logger.warning(f"Spectral gate failed ({exc}) — skipping")
        return False
