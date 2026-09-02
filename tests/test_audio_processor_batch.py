"""
tests/test_audio_processor_batch.py
Unit tests for the two new AudioProcessor methods added for the Sarvam
Batch STT path — to_wav() (conversion only, no chunking) and
split_for_batch() (reuses the existing proven ffmpeg segment mechanism,
only used for recordings that exceed the Batch API's ~2h/file limit).
Uses real ffmpeg (available in this environment) against a tiny synthetic
silent WAV generated with Python's stdlib `wave` module — no network,
fast, no external fixtures needed.
"""
import struct
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from audio.processor import AudioProcessor


def _write_silent_wav(path: Path, seconds: float, sample_rate: int = 16000) -> None:
    n_frames = int(seconds * sample_rate)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n_frames}h", *([0] * n_frames)))


def test_to_wav_converts_and_caches(tmp_path):
    src = tmp_path / "input.wav"
    _write_silent_wav(src, seconds=2.0)

    processor = AudioProcessor()
    out1 = processor.to_wav(src)
    assert out1.exists()
    mtime1 = out1.stat().st_mtime

    # Second call should hit the existing-file cache path (_convert_to_wav
    # returns early if out_path already exists) rather than re-running ffmpeg.
    out2 = processor.to_wav(src)
    assert out2 == out1
    assert out2.stat().st_mtime == mtime1


def test_split_for_batch_produces_expected_segment_count(tmp_path):
    src = tmp_path / "input.wav"
    _write_silent_wav(src, seconds=5.0)

    processor = AudioProcessor()
    wav_path = processor.to_wav(src)

    chunks = processor.split_for_batch(wav_path, max_seconds=2.0)

    # 5s split into <=2s pieces -> 3 segments (2s, 2s, 1s)
    assert len(chunks) == 3
    # Segments are contiguous and in order, each within max_seconds (+ small tolerance)
    cursor = 0.0
    for c in chunks:
        assert c.start_time >= cursor - 0.05
        assert (c.end_time - c.start_time) <= 2.05
        assert Path(c.file_path).exists()
        cursor = c.end_time
    assert cursor >= 4.9   # covers ~the whole 5s recording


def test_split_for_batch_single_segment_when_under_limit(tmp_path):
    src = tmp_path / "input.wav"
    _write_silent_wav(src, seconds=2.0)

    processor = AudioProcessor()
    wav_path = processor.to_wav(src)

    chunks = processor.split_for_batch(wav_path, max_seconds=7200.0)
    assert len(chunks) == 1
