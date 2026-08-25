"""
diarization/diarizer_v2.py
Production speaker diarization with talk analytics.

Strategy priority:
  1. Pyannote 3.1 (if HF_TOKEN set and pyannote installed)
  2. Stereo channel assignment (L=Salesperson, R=Customer)
  3. Energy-based turn detection (mono fallback)

Also computes TalkAnalytics (talk ratio, interruptions, monologue detection).
"""
from __future__ import annotations
import logging
import subprocess
import tempfile
import wave
import struct
from pathlib import Path

from config.models_multilingual import (
    AudioChunkV2, SpeakerRole, SpeakerSegment, TalkAnalytics
)
from config.settings import get_settings

logger   = logging.getLogger(__name__)
settings = get_settings()

STEREO_RATIO_MIN  = 1.20   # L must be ≥20% louder to assign to salesperson
TURN_GAP_SEC      = 0.4    # silence gap that indicates speaker change (mono)
FRAME_SEC         = 0.2    # analysis frame size for energy detection
MONOLOGUE_SEC     = 45.0   # monologue threshold


class DiarizationResult:
    def __init__(
        self,
        speaker_map: dict[int, SpeakerRole],          # chunk_id → speaker
        segments:    list[SpeakerSegment],
        analytics:   TalkAnalytics,
    ):
        self.speaker_map = speaker_map
        self.segments    = segments
        self.analytics   = analytics


class DiarizationEngineV2:

    def diarize(self, wav_path: Path, chunks: list[AudioChunkV2]) -> DiarizationResult:
        diarization_provider = getattr(settings, "diarization_provider", "auto")

        if diarization_provider == "pyannote" or (
            diarization_provider == "auto" and getattr(settings, "hf_token", "")
        ):
            try:
                return self._diarize_pyannote(wav_path, chunks)
            except Exception as exc:
                logger.warning(f"Pyannote diarization failed ({exc}), trying next strategy")

        n_channels = _get_channels(wav_path)
        if n_channels >= 2:
            logger.info("Stereo audio → channel-based diarization")
            return self._diarize_stereo(wav_path, chunks)

        logger.info("Mono audio → energy turn-detection diarization")
        return self._diarize_mono_energy(wav_path, chunks)

    # ── Pyannote 3.1 ──────────────────────────────────────────────────────────

    def _diarize_pyannote(self, wav_path: Path, chunks: list[AudioChunkV2]) -> DiarizationResult:
        from pyannote.audio import Pipeline
        import torch

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=settings.hf_token,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline.to(torch.device(device))
        logger.info(f"Running Pyannote diarization on {device}…")

        diarization = pipeline(str(wav_path))

        # Map speaker labels to roles (first speaker = salesperson heuristic)
        speaker_first_seen: dict[str, float] = {}
        raw_segments: list[tuple[float, float, str]] = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if speaker not in speaker_first_seen:
                speaker_first_seen[speaker] = turn.start
            raw_segments.append((turn.start, turn.end, speaker))

        # Sort speakers by first appearance; first = salesperson
        ordered = sorted(speaker_first_seen.items(), key=lambda x: x[1])
        role_map: dict[str, SpeakerRole] = {}
        for i, (spk, _) in enumerate(ordered):
            role_map[spk] = SpeakerRole.SALESPERSON if i == 0 else SpeakerRole.CUSTOMER

        segments = [
            SpeakerSegment(
                speaker=role_map.get(spk, SpeakerRole.UNKNOWN),
                start=s, end=e, duration=round(e - s, 3)
            )
            for s, e, spk in raw_segments
        ]

        speaker_map = _segments_to_chunk_map(segments, chunks)
        analytics   = _compute_analytics(segments, chunks)
        return DiarizationResult(speaker_map, segments, analytics)

    # ── Stereo channel diarization ─────────────────────────────────────────────

    def _diarize_stereo(self, wav_path: Path, chunks: list[AudioChunkV2]) -> DiarizationResult:
        """Extract L/R channels; compare per-chunk RMS energy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            left  = Path(tmpdir) / "left.wav"
            right = Path(tmpdir) / "right.wav"
            _extract_channel(wav_path, 0, left)
            _extract_channel(wav_path, 1, right)

            segments: list[SpeakerSegment] = []
            for chunk in chunks:
                l_rms = _read_rms(left,  chunk.start_time, chunk.end_time)
                r_rms = _read_rms(right, chunk.start_time, chunk.end_time)

                if l_rms > r_rms * STEREO_RATIO_MIN:
                    speaker = SpeakerRole.SALESPERSON
                elif r_rms > l_rms * STEREO_RATIO_MIN:
                    speaker = SpeakerRole.CUSTOMER
                else:
                    # Roughly equal — pick by ratio
                    speaker = SpeakerRole.SALESPERSON if l_rms >= r_rms else SpeakerRole.CUSTOMER

                segments.append(SpeakerSegment(
                    speaker=speaker,
                    start=chunk.start_time,
                    end=chunk.end_time,
                    duration=round(chunk.end_time - chunk.start_time, 3),
                ))

        speaker_map = {c.chunk_id: segments[i].speaker for i, c in enumerate(chunks)}
        analytics   = _compute_analytics(segments, chunks)
        return DiarizationResult(speaker_map, segments, analytics)

    # ── Energy turn detection (mono) ───────────────────────────────────────────

    def _diarize_mono_energy(self, wav_path: Path, chunks: list[AudioChunkV2]) -> DiarizationResult:
        """
        Heuristic: alternate speakers at silence gaps > TURN_GAP_SEC.
        First speaker assumed to be salesperson (they usually initiate).
        """
        segments: list[SpeakerSegment] = []
        current_speaker = SpeakerRole.SALESPERSON
        prev_end = 0.0

        for chunk in sorted(chunks, key=lambda c: c.start_time):
            gap = chunk.start_time - prev_end
            if gap > TURN_GAP_SEC and segments:
                current_speaker = (
                    SpeakerRole.CUSTOMER
                    if current_speaker == SpeakerRole.SALESPERSON
                    else SpeakerRole.SALESPERSON
                )
            segments.append(SpeakerSegment(
                speaker=current_speaker,
                start=chunk.start_time,
                end=chunk.end_time,
                duration=round(chunk.end_time - chunk.start_time, 3),
            ))
            prev_end = chunk.end_time

        speaker_map = {c.chunk_id: segments[i].speaker for i, c in enumerate(chunks)}
        analytics   = _compute_analytics(segments, chunks)
        return DiarizationResult(speaker_map, segments, analytics)


# ── Analytics ──────────────────────────────────────────────────────────────────

def _compute_analytics(
    segments: list[SpeakerSegment],
    chunks:   list[AudioChunkV2],
) -> TalkAnalytics:
    total = sum(s.duration for s in segments) or 1.0

    sp_dur = sum(s.duration for s in segments if s.speaker == SpeakerRole.SALESPERSON)
    cu_dur = sum(s.duration for s in segments if s.speaker == SpeakerRole.CUSTOMER)

    # Interruptions: speaker changes within <0.3s of previous end
    interruptions = 0
    sorted_segs   = sorted(segments, key=lambda s: s.start)
    for i in range(1, len(sorted_segs)):
        prev = sorted_segs[i - 1]
        curr = sorted_segs[i]
        if curr.speaker != prev.speaker and curr.start - prev.end < 0.3:
            interruptions += 1

    # Longest monologue
    longest = max((s.duration for s in segments), default=0.0)

    # Dead air (gaps between segments)
    dead_air = 0.0
    for i in range(1, len(sorted_segs)):
        gap = sorted_segs[i].start - sorted_segs[i - 1].end
        if gap > 0.5:
            dead_air += gap

    # Avg response latency (time between turns)
    latencies = []
    for i in range(1, len(sorted_segs)):
        if sorted_segs[i].speaker != sorted_segs[i - 1].speaker:
            lat = sorted_segs[i].start - sorted_segs[i - 1].end
            if 0 < lat < 10:
                latencies.append(lat)
    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0

    return TalkAnalytics(
        salesperson_pct=round(sp_dur / total * 100, 1),
        customer_pct=round(cu_dur / total * 100, 1),
        salesperson_duration=round(sp_dur, 1),
        customer_duration=round(cu_dur, 1),
        interruption_count=interruptions,
        longest_monologue_sec=round(longest, 1),
        avg_response_latency=round(avg_lat, 2),
        dead_air_sec=round(dead_air, 1),
        questions_asked=0,   # filled in by post-processor
    )


# ── Utilities ──────────────────────────────────────────────────────────────────

def _segments_to_chunk_map(
    segments: list[SpeakerSegment], chunks: list[AudioChunkV2]
) -> dict[int, SpeakerRole]:
    result = {}
    for chunk in chunks:
        mid = (chunk.start_time + chunk.end_time) / 2
        speaker = SpeakerRole.UNKNOWN
        for seg in segments:
            if seg.start <= mid <= seg.end:
                speaker = seg.speaker
                break
        result[chunk.chunk_id] = speaker
    return result


def _get_channels(wav_path: Path) -> int:
    try:
        with wave.open(str(wav_path), "rb") as wf:
            return wf.getnchannels()
    except Exception:
        return 1


def _extract_channel(src: Path, channel: int, dst: Path) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-filter_complex", f"pan=mono|c0=c{channel}",
        "-ar", "16000", "-c:a", "pcm_s16le", str(dst),
    ]
    subprocess.run(cmd, capture_output=True, check=False)


def _read_rms(wav_path: Path, start: float, end: float) -> float:
    try:
        with wave.open(str(wav_path), "rb") as wf:
            sr  = wf.getframerate()
            sw  = wf.getsampwidth()
            nf  = wf.getnframes()
            sf  = min(int(start * sr), nf)
            ef  = min(int(end * sr), nf)
            wf.setpos(sf)
            raw = wf.readframes(ef - sf)
        fmt = {1: "b", 2: "h", 4: "i"}.get(sw, "h")
        n = (ef - sf)
        samples = struct.unpack(f"<{n}{fmt}", raw[:n * sw])
        return (sum(s * s for s in samples) / max(len(samples), 1)) ** 0.5
    except Exception:
        return 0.0
