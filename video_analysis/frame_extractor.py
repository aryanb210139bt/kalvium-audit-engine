"""
video_analysis/frame_extractor.py
Duration/metadata probing + single-frame extraction via FFmpeg. Never decodes
the whole video — each frame is grabbed with `-ss` (seek) before `-i` (input),
which lets FFmpeg jump straight to the timestamp instead of scanning. Purely
local: FFmpeg/FFprobe/Pillow only, no AI API, no GPU.
"""
from __future__ import annotations
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# Percentage-based sampling points — works for any call length without
# hardcoded timestamps. Deliberately avoids exactly 0%/100%: the very start
# is often a black/loading frame, the very end is often frozen/post-call UI.
SAMPLE_PERCENTAGES = [5, 25, 50, 75, 95]
MIN_DURATION_SECONDS = 5.0  # below this, sample points would collide

# Frame validation: retry a nearby timestamp if the grabbed frame looks
# blank/black — cheap local pixel check, no AI model involved.
VALIDATION_RETRY_OFFSETS_SECONDS = [5, -5, 10, -10]
BLACK_FRAME_MEAN_THRESHOLD = 8.0   # 0-255 mean brightness below this = "black"
BLACK_FRAME_STDDEV_THRESHOLD = 4.0  # near-zero variance = flat/blank frame


def probe_duration(path: Path) -> float:
    """Get duration (seconds) of a video/audio file via ffprobe. 0.0 on failure."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet",
             "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1",
             str(path)],
            capture_output=True, text=True, check=True,
        )
        return float(result.stdout.strip())
    except Exception as e:
        logger.warning(f"probe_duration failed for {path}: {e}")
        return 0.0


def probe_metadata(path: Path) -> dict:
    """
    Full local metadata via a single ffprobe call — duration, resolution,
    fps, codecs, sample rate, channels, plus file size from the filesystem.
    Never calls out to any API. Returns a dict with whatever it could read;
    missing/unreadable fields are omitted rather than raising.
    """
    meta: dict = {}
    try:
        meta["file_size_bytes"] = path.stat().st_size
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", "-show_streams", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        import json
        data = json.loads(result.stdout or "{}")
        fmt = data.get("format", {})
        if fmt.get("duration"):
            meta["duration_seconds"] = float(fmt["duration"])

        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video" and "width" not in meta:
                meta["width"] = stream.get("width")
                meta["height"] = stream.get("height")
                meta["video_codec"] = stream.get("codec_name")
                rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or ""
                if "/" in rate:
                    num, den = rate.split("/")
                    if float(den or 0) > 0:
                        meta["fps"] = round(float(num) / float(den), 2)
            elif stream.get("codec_type") == "audio" and "audio_codec" not in meta:
                meta["audio_codec"] = stream.get("codec_name")
                meta["sample_rate"] = int(stream["sample_rate"]) if stream.get("sample_rate") else None
                meta["channels"] = stream.get("channels")
    except Exception as e:
        logger.warning(f"probe_metadata failed for {path}: {e}")

    return meta


def compute_timestamps(duration: float, percentages: list[int] | None = None) -> list[dict]:
    """
    Return {"label", "percentage", "seconds"} sample points at the given
    percentages of total duration (default SAMPLE_PERCENTAGES: 5/25/50/75/95).
    Works for any call length — no hardcoded timestamps. For very short
    videos, near-duplicate points collapse to avoid redundant frames.
    """
    if duration <= 0:
        return []

    pcts = percentages or SAMPLE_PERCENTAGES
    points = []
    seen_seconds = set()
    for pct in pcts:
        t = duration * (pct / 100.0)
        t = max(0.0, min(t, duration - 0.1))
        rounded = round(t, 1)
        if rounded in seen_seconds:
            continue  # video too short — avoid duplicate/near-duplicate frames
        seen_seconds.add(rounded)
        points.append({"label": f"{pct}%", "percentage": pct, "seconds": rounded})
    return points


MAX_SCREENSHOT_LONG_EDGE = 960  # px — plenty to identify UI tiles/slides;
                                # keeps stored evidence + vision-model payload
                                # small regardless of the source video's
                                # actual resolution.


def _downscale_in_place(path: Path, long_edge: int = MAX_SCREENSHOT_LONG_EDGE) -> None:
    """Resize + recompress a JPEG in place if it's larger than `long_edge` on
    its longest side. Uses Pillow (already a transitive dependency via
    reportlab — no new package added). Never raises; a failure here just
    leaves the original ffmpeg output in place."""
    try:
        from PIL import Image
        with Image.open(path) as img:
            w, h = img.size
            longest = max(w, h)
            if longest <= long_edge:
                return
            scale = long_edge / longest
            img = img.convert("RGB").resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
            img.save(path, "JPEG", quality=80, optimize=True)
    except Exception as e:
        logger.warning(f"_downscale_in_place failed for {path}: {e}")


def _is_frame_valid(path: Path) -> bool:
    """
    Cheap local sanity check — no AI model. Rejects frames that are
    essentially solid black or a single flat color (a common artifact of
    seeking to a keyframe boundary during a scene transition, or landing on
    a loading/blank screen). Uses Pillow's built-in histogram; a corrupt or
    zero-byte file also fails this check.
    """
    try:
        from PIL import Image, ImageStat
        with Image.open(path) as img:
            gray = img.convert("L")
            stat = ImageStat.Stat(gray)
            mean = stat.mean[0]
            stddev = stat.stddev[0]
            if mean < BLACK_FRAME_MEAN_THRESHOLD and stddev < BLACK_FRAME_STDDEV_THRESHOLD:
                return False
            return True
    except Exception as e:
        logger.warning(f"_is_frame_valid check failed for {path}: {e}")
        return False  # unreadable/corrupt file — treat as invalid


def extract_frame(source: Path, timestamp_seconds: float, out_path: Path) -> bool:
    """
    Grab a single JPEG frame from `source` at `timestamp_seconds`, resized/
    compressed to MAX_SCREENSHOT_LONG_EDGE. Returns True on success. Never
    raises — caller treats a failed frame as a gap in coverage, not a fatal
    error (one bad frame must not sink the other four).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(timestamp_seconds),
        "-i", str(source),
        "-frames:v", "1",
        "-q:v", "3",
        str(out_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0 or not out_path.exists():
            logger.warning(f"extract_frame failed at {timestamp_seconds}s: {result.stderr[-400:]}")
            return False
        _downscale_in_place(out_path)
        return True
    except Exception as e:
        logger.warning(f"extract_frame error at {timestamp_seconds}s: {e}")
        return False


def extract_frame_validated(source: Path, timestamp_seconds: float, out_path: Path,
                             duration: float) -> tuple[bool, float]:
    """
    extract_frame() plus a local black/blank-frame check with retry at
    nearby timestamps (no AI model — see _is_frame_valid). Returns
    (success, actual_timestamp_used). Tries the requested timestamp first,
    then VALIDATION_RETRY_OFFSETS_SECONDS in order, clamped to the video's
    actual bounds. Still never raises — exhausting all retries just means
    this frame is reported as failed, same as a plain extraction failure.
    """
    candidates = [timestamp_seconds] + [
        timestamp_seconds + off for off in VALIDATION_RETRY_OFFSETS_SECONDS
    ]
    for t in candidates:
        t = max(0.0, min(t, max(duration - 0.1, 0.0)))
        if extract_frame(source, t, out_path):
            if _is_frame_valid(out_path):
                return True, t
            logger.info(f"Frame at {t}s looked black/blank — trying a nearby timestamp")
    return False, timestamp_seconds
