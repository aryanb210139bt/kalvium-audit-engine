"""
audio/processor.py
Convert any audio/video to 16kHz WAV and split into ≤28s chunks.
Uses only ffmpeg (subprocess) — no pydub/audioop dependency.
Python 3.13 compatible.
"""
from __future__ import annotations
import json
import logging
import subprocess
import struct
import wave
from pathlib import Path

from config.models import AudioChunk

logger = logging.getLogger(__name__)

TARGET_SR     = 16000
CHUNK_MAX_SEC = 28      # Sarvam AI limit is 30s; 28s gives 2s headroom — now only
                        # a _split_chunks() default; the one active caller
                        # (split_for_batch) always passes its own max_seconds


# Sarvam Batch STT's documented hard limit is 2h/file; BATCH_MAX_SEC (see
# transcription/sarvam_batch.BATCH_MAX_SECONDS, kept in one place there)
# leaves a safety margin. split_for_batch() below reuses the exact same
# proven ffmpeg segment mechanism as _split_chunks(), just parameterised —
# no new splitting logic, only a different segment length.

class AudioProcessor:

    def to_wav(self, recording_path: Path) -> Path:
        """
        Batch STT path: convert to 16kHz WAV only — no chunking. The whole
        file is handed to Sarvam's Batch API as a single job (or split into
        <=~2h segments by split_for_batch, only for recordings that exceed
        the Batch API's per-file limit).
        """
        return self._convert_to_wav(recording_path)

    def split_for_batch(self, wav_path: Path, max_seconds: float) -> list[AudioChunk]:
        """
        Only used for recordings longer than the Batch API's per-file limit
        (~2h). Splits into the fewest possible <=max_seconds segments (e.g.
        2 segments for a 3-hour file), each becoming its own Batch STT job
        in pipeline_v3._batch_stt. Reuses _split_chunks' exact ffmpeg
        segment/reset_timestamps mechanism — only the segment length
        differs from the legacy 28s chunking.
        """
        return self._split_chunks(wav_path, max_sec=max_seconds, subdir="batch_segments")

    def get_duration(self, wav_path: Path) -> float:
        return _ffprobe_duration(wav_path)

    def get_channels(self, wav_path: Path) -> int:
        """Return number of audio channels (1=mono, 2=stereo)."""
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json",
                 "-show_streams", str(wav_path)],
                capture_output=True, text=True, check=True
            )
            info = json.loads(result.stdout)
            for stream in info.get("streams", []):
                if stream.get("codec_type") == "audio":
                    return int(stream.get("channels", 1))
        except Exception:
            pass
        return 1

    # ── Conversion ─────────────────────────────────────────────────────────────

    def _convert_to_wav(self, input_path: Path) -> Path:
        """Convert any media file to 16kHz WAV, preserving original channel count."""
        out_path = input_path.parent / f"{input_path.stem}_16k.wav"
        if out_path.exists():
            logger.info(f"  WAV cache hit: {out_path.name}")
            return out_path
        cmd = [
            "ffmpeg", "-y", "-i", str(input_path),
            "-ar", str(TARGET_SR),
            # Keep original channel count so diarizer can read stereo
            "-f", "wav", str(out_path),
        ]
        logger.info(f"  Converting to WAV: {input_path.name}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg conversion failed:\n{result.stderr[:600]}")
        return out_path

    # ── Chunking ───────────────────────────────────────────────────────────────

    def _split_chunks(self, wav_path: Path, max_sec: float = CHUNK_MAX_SEC,
                       subdir: str = "chunks") -> list[AudioChunk]:
        """
        Split WAV into <=max_sec mono chunks using ffmpeg segment filter.
        Each chunk is exported as a separate mono WAV. Default max_sec/subdir
        preserve the original <=28s legacy chunking behavior exactly;
        split_for_batch() reuses this with a ~2h max_sec for the rare
        >2h-recording Batch STT case instead.
        """
        chunk_dir = wav_path.parent / subdir
        chunk_dir.mkdir(exist_ok=True)

        duration = _ffprobe_duration(wav_path)
        if duration <= 0:
            raise RuntimeError(f"Could not determine audio duration: {wav_path}")

        logger.info(f"  Audio duration: {duration:.1f}s, splitting into ≤{max_sec:.0f}s chunks")

        # Use ffmpeg segment to split — forces mono output for STT compatibility
        pattern = str(chunk_dir / "chunk_%04d.wav")
        cmd = [
            "ffmpeg", "-y", "-i", str(wav_path),
            "-ac", "1",                          # mono for STT
            "-ar", str(TARGET_SR),
            "-f", "segment",
            "-segment_time", str(max_sec),
            "-reset_timestamps", "1",
            "-c:a", "pcm_s16le",
            pattern,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg segment failed:\n{result.stderr[:600]}")

        # Collect generated chunk files in order
        chunk_files = sorted(chunk_dir.glob("chunk_*.wav"))
        if not chunk_files:
            raise RuntimeError("ffmpeg produced no chunk files")

        chunks: list[AudioChunk] = []
        cursor = 0.0
        for i, cf in enumerate(chunk_files):
            ch_dur = _ffprobe_duration(cf)
            chunks.append(AudioChunk(
                chunk_id=i,
                start_time=round(cursor, 3),
                end_time=round(cursor + ch_dur, 3),
                file_path=str(cf),
            ))
            cursor += ch_dur

        max_dur = max(c.end_time - c.start_time for c in chunks)
        logger.info(
            f"  {len(chunks)} chunks, max={max_dur:.1f}s, total={cursor:.1f}s"
        )
        return chunks


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ffprobe_duration(path: Path) -> float:
    """Get duration of an audio/video file via ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet",
             "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1",
             str(path)],
            capture_output=True, text=True, check=True,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def read_wav_rms(wav_path: Path, start_sec: float = 0, end_sec: float = -1) -> float:
    """
    Read a WAV file and compute RMS amplitude of a time slice.
    Used by diarizer for energy-based speaker detection.
    Pure stdlib (wave + struct) — no pydub needed.
    """
    try:
        with wave.open(str(wav_path), "rb") as wf:
            n_channels  = wf.getnchannels()
            sampwidth   = wf.getsampwidth()
            framerate   = wf.getframerate()
            n_frames    = wf.getnframes()

            start_frame = int(start_sec * framerate)
            end_frame   = int(end_sec * framerate) if end_sec >= 0 else n_frames
            end_frame   = min(end_frame, n_frames)

            wf.setpos(start_frame)
            frames_to_read = max(0, end_frame - start_frame)
            if frames_to_read == 0:
                return 0.0
            raw = wf.readframes(frames_to_read)

        fmt   = {1: "b", 2: "h", 4: "i"}.get(sampwidth, "h")
        total = n_channels * frames_to_read
        samples = struct.unpack(f"<{total}{fmt}", raw[:total * sampwidth])
        sq_sum = sum(s * s for s in samples)
        return (sq_sum / max(1, len(samples))) ** 0.5
    except Exception:
        return 0.0


def extract_channel_wav(src: Path, channel: int, dest: Path) -> None:
    """
    Extract a single channel from a stereo WAV using ffmpeg.
    channel: 0 = left (counsellor), 1 = right (student)
    """
    # ffmpeg pan filter: "mono|c0=c{channel}"
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-filter_complex", f"pan=mono|c0=c{channel}",
        "-ar", str(TARGET_SR),
        "-c:a", "pcm_s16le",
        str(dest),
    ]
    subprocess.run(cmd, capture_output=True, check=True)
