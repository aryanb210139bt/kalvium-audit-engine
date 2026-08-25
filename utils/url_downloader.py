"""
utils/url_downloader.py
Download meeting recordings from URLs.

Supported sources:
  - Google Drive  (drive.google.com/file/d/<ID>/view)
  - Loom          (loom.com/share/<ID>)
  - Direct links  (.mp4 / .mp3 / .wav / .m4a / .webm / .mov / .ogg)
  - Any HTTP/HTTPS direct-download URL

Storage strategy (V1):
  download_as_audio() pipes the HTTP response directly into FFmpeg which
  writes only an MP3 — the raw video never lands on disk.
  Peak disk use per recording ≈ 100 MB regardless of source file size.
"""
from __future__ import annotations
import io
import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".mp4", ".mp3", ".wav", ".m4a", ".ogg", ".webm", ".mov"}
MAX_DOWNLOAD_BYTES = 2 * 1024 * 1024 * 1024   # 2 GB — covers 2-hour Google Meet recordings
CHUNK_SIZE = 8 * 1024 * 1024                   # 8 MB streaming chunks
TIMEOUT_CONNECT = 30
TIMEOUT_READ    = 600                           # 10 min — large files on slow networks

_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
})


# ── URL classifiers ───────────────────────────────────────────────────────────

def _gdrive_id(url: str) -> Optional[str]:
    patterns = [
        r"drive\.google\.com/file/d/([a-zA-Z0-9_-]+)",
        r"drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)",
        r"drive\.google\.com/uc\?.*[?&]id=([a-zA-Z0-9_-]+)",
        r"docs\.google\.com/.*[?&]id=([a-zA-Z0-9_-]+)",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None


def _loom_id(url: str) -> Optional[str]:
    m = re.search(r"loom\.com/share/([a-zA-Z0-9]+)", url)
    return m.group(1) if m else None


def detect_source(url: str) -> str:
    """Returns 'gdrive' | 'loom' | 'direct'."""
    if _gdrive_id(url):
        return "gdrive"
    if _loom_id(url):
        return "loom"
    return "direct"


# ── Downloaders ───────────────────────────────────────────────────────────────

def _download_google_drive(file_id: str, dest_dir: Path) -> Path:
    """
    Download a file from Google Drive via gdown, which handles the "file too
    large for Google to virus-scan" confirmation interstitial internally.

    This used to be a hand-rolled requests-based parser of that interstitial
    page (looking for a `confirm=` token or a `downloadForm` element) — Google
    has since changed that page's markup (hyphenated `download-form` id, the
    confirm token now a separate hidden `<input>` rather than embedded in the
    URL), which silently broke both regexes and made every large (~100MB+)
    file fail with a misleading "check your sharing settings" error even when
    sharing was correct. gdown is already a project dependency and already
    handles this correctly for the audio-only path (download_as_audio below)
    — reusing it here instead of maintaining a second, now-stale parser.
    """
    try:
        import gdown
    except ImportError:
        raise ValueError(
            "gdown is required for Google Drive downloads. "
            "Install with: pip install gdown"
        )

    logger.info(f"Google Drive download (gdown): id={file_id}")
    dest = dest_dir / "recording.mp4"
    try:
        gdown.download(id=file_id, output=str(dest), quiet=False)
    except Exception as e:
        raise ValueError(
            f"Google Drive download failed: {e}. "
            "Ensure the file is shared as 'Anyone with the link can view'."
        )

    if not dest.exists() or dest.stat().st_size < 10_000:
        dest.unlink(missing_ok=True)
        raise ValueError(
            "Google Drive returned an empty or invalid file. "
            "Check that: (1) the file exists, (2) sharing is set to "
            "'Anyone with the link can view', (3) the file ID in the URL is correct."
        )

    if dest.stat().st_size > MAX_DOWNLOAD_BYTES:
        size_mb = dest.stat().st_size // 1024 // 1024
        dest.unlink(missing_ok=True)
        raise ValueError(
            f"Download is {size_mb}MB, exceeds the {MAX_DOWNLOAD_BYTES // 1024 // 1024}MB limit"
        )

    return dest


def _download_loom(video_id: str, dest_dir: Path) -> Path:
    """
    Download a Loom video via their API endpoint.
    Note: Loom's API requires authentication for private videos.
    Public Loom share links can be downloaded by fetching the MP4 URL from the page.
    """
    share_url = f"https://www.loom.com/share/{video_id}"
    logger.info(f"Loom download: id={video_id}")

    # Fetch share page to extract direct MP4 URL
    page = _SESSION.get(share_url, timeout=(TIMEOUT_CONNECT, 60))
    page.raise_for_status()
    html = page.text

    # Look for the MP4 download URL embedded in the page
    mp4_match = re.search(r'"url"\s*:\s*"(https://cdn\.loom\.com/[^"]+\.mp4[^"]*)"', html)
    if not mp4_match:
        mp4_match = re.search(r'(https://cdn\.loom\.com/[^"\s]+\.mp4[^"\s]*)', html)
    if not mp4_match:
        raise ValueError(
            "Could not extract MP4 URL from Loom page. "
            "Ensure the Loom video is publicly shared."
        )

    mp4_url = mp4_match.group(1)
    logger.info(f"Loom MP4 URL found: {mp4_url[:80]}…")
    return _download_direct_url(mp4_url, dest_dir)


def _download_direct_url(url: str, dest_dir: Path) -> Path:
    """Download any direct HTTP/HTTPS URL."""
    logger.info(f"Direct download: {url[:80]}…")
    resp = _SESSION.get(url, stream=True,
                        timeout=(TIMEOUT_CONNECT, TIMEOUT_READ), allow_redirects=True)
    resp.raise_for_status()
    ext = _ext_from_response(resp) or _ext_from_url(url) or ".mp4"
    dest = dest_dir / f"recording{ext}"
    _stream_to_file(resp, dest)
    return dest


def _stream_to_file(resp: requests.Response, dest: Path) -> None:
    """Stream response body to a file, enforcing max size."""
    downloaded = 0
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                downloaded += len(chunk)
                if downloaded > MAX_DOWNLOAD_BYTES:
                    raise ValueError(
                        f"Download exceeds {MAX_DOWNLOAD_BYTES // 1024 // 1024}MB limit"
                    )
                f.write(chunk)
    logger.info(f"Downloaded {downloaded // 1024}KB → {dest.name}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ext_from_response(resp: requests.Response) -> Optional[str]:
    """Try to determine file extension from response headers."""
    cd = resp.headers.get("Content-Disposition", "")
    m = re.search(r'filename[^;=\n]*=["\']?([^"\';\n]+)', cd)
    if m:
        name = m.group(1).strip().strip('"\'')
        ext = Path(name).suffix.lower()
        if ext in ALLOWED_EXTENSIONS:
            return ext

    ct = resp.headers.get("Content-Type", "").split(";")[0].strip()
    ct_map = {
        "video/mp4": ".mp4",
        "audio/mpeg": ".mp3",
        "audio/mp4": ".m4a",
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/ogg": ".ogg",
        "video/webm": ".webm",
        "audio/webm": ".webm",
        "video/quicktime": ".mov",
    }
    return ct_map.get(ct)


def _ext_from_url(url: str) -> Optional[str]:
    """Extract file extension from the URL path."""
    path = urlparse(url).path
    ext = Path(path).suffix.lower()
    return ext if ext in ALLOWED_EXTENSIONS else None


# ── Public API ────────────────────────────────────────────────────────────────

def download_recording(url: str, dest_dir: Path) -> Path:
    """
    Download a meeting recording from any supported URL.

    Args:
        url:      Source URL (Google Drive, Loom, or direct link)
        dest_dir: Directory to save the file into

    Returns:
        Path to the downloaded file

    Raises:
        ValueError:  Unsupported URL, too large, unsupported format
        requests.HTTPError: Network / HTTP error
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    url = url.strip()

    source = detect_source(url)

    if source == "gdrive":
        file_id = _gdrive_id(url)
        return _download_google_drive(file_id, dest_dir)

    if source == "loom":
        video_id = _loom_id(url)
        return _download_loom(video_id, dest_dir)

    # Direct / unknown — validate extension before downloading
    ext = _ext_from_url(url)
    if ext and ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}")

    return _download_direct_url(url, dest_dir)


def validate_url(url: str) -> None:
    """
    Quick HEAD check to verify the URL is reachable.
    Raises ValueError with a user-friendly message on failure.
    """
    url = url.strip()
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("URL must start with http:// or https://")

    # Skip HEAD check for Google Drive (redirects confuse it)
    if detect_source(url) == "gdrive":
        return

    try:
        r = _SESSION.head(url, timeout=(10, 10), allow_redirects=True)
        r.raise_for_status()
    except requests.RequestException as e:
        raise ValueError(f"URL not reachable: {e}") from e


def preflight_drive(url: str) -> None:
    """
    Verify a Google Drive URL is publicly accessible (returns a file, not a permission error).
    Raises ValueError with a user-friendly message if the file is restricted.
    """
    if detect_source(url) != "gdrive":
        return
    file_id = _gdrive_id(url)
    if not file_id:
        raise ValueError("Could not extract Google Drive file ID from URL")
    try:
        check_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        r = _SESSION.head(check_url, timeout=(15, 15), allow_redirects=True)
        # A 403 here means private / restricted sharing
        if r.status_code == 403:
            raise ValueError(
                "Google Drive file is not publicly accessible. "
                "Please set sharing to 'Anyone with the link can view'."
            )
    except requests.RequestException as e:
        logger.warning(f"Drive preflight check failed (non-fatal): {e}")


def _loom_direct_url(video_id: str) -> str:
    """Resolve a Loom share ID to its CDN MP4 URL."""
    page = _SESSION.get(f"https://www.loom.com/share/{video_id}",
                        timeout=(TIMEOUT_CONNECT, 60))
    page.raise_for_status()
    mp4_m = re.search(r'"url"\s*:\s*"(https://cdn\.loom\.com/[^"]+\.mp4[^"]*)"', page.text)
    if not mp4_m:
        mp4_m = re.search(r'(https://cdn\.loom\.com/[^"\s]+\.mp4[^"\s]*)', page.text)
    if not mp4_m:
        raise ValueError("Could not find Loom MP4 URL. Ensure the video is publicly shared.")
    return mp4_m.group(1)


def extract_audio_from_local_file(video_path: Path, mp3_path: Path) -> None:
    """
    Public alias for _ffmpeg_extract_mp3, for the video-preprocessing stage
    (video_analysis/) to extract audio from an already-downloaded local
    video — reusing this exact extraction rather than a second, duplicate
    implementation. Same mono 16kHz MP3 output the rest of the app expects.
    """
    _ffmpeg_extract_mp3(str(video_path), mp3_path)


def _ffmpeg_extract_mp3(input_path_or_url: str, mp3_path: Path,
                        timeout: int = TIMEOUT_READ) -> None:
    """
    Run FFmpeg to extract audio as mono 16kHz MP3.
    input_path_or_url can be a local file path or an HTTP(S) URL.
    """
    is_http = input_path_or_url.startswith("http://") or input_path_or_url.startswith("https://")

    ffmpeg_cmd = ["ffmpeg", "-y"]
    if is_http:
        # -headers is an HTTP demuxer option — only valid for network inputs
        ffmpeg_cmd += ["-headers", "User-Agent: Mozilla/5.0\r\n"]
    ffmpeg_cmd += [
        "-i", input_path_or_url,
        "-vn",
        "-acodec", "libmp3lame",
        "-q:a", "2",       # VBR ~190 kbps
        "-ac", "1",        # mono
        "-ar", "16000",    # 16 kHz for Whisper
        str(mp3_path),
    ]
    result = subprocess.run(ffmpeg_cmd, capture_output=True, timeout=timeout)
    if result.returncode != 0:
        # Skip the long FFmpeg banner and show only the actual error lines
        full_stderr = result.stderr.decode("utf-8", errors="ignore")
        error_lines = [l for l in full_stderr.splitlines()
                       if any(k in l for k in ("Error", "error", "Invalid", "No such", "failed", "cannot", "403", "Forbidden", "not found"))]
        stderr = "\n".join(error_lines[-10:]) or full_stderr[-400:]
        if "403" in stderr or "Forbidden" in stderr:
            raise ValueError(
                "Server returned 403 Forbidden. "
                "For Google Drive: set sharing to 'Anyone with the link can view'."
            )
        raise RuntimeError(
            f"FFmpeg audio extraction failed (exit {result.returncode}):\n{stderr}"
        )
    if not mp3_path.exists() or mp3_path.stat().st_size < 1000:
        raise RuntimeError(
            "FFmpeg produced no audio output — source file may be empty or corrupt."
        )


def download_as_audio(url: str, dest_dir: Path,
                      progress_cb=None) -> Path:
    """
    Download a recording and extract audio as MP3.

    Strategy per source:
      - Google Drive → gdown downloads the video to a temp file → FFmpeg extracts MP3
                       → temp video deleted immediately (only MP3 kept)
      - Loom         → resolve CDN MP4 URL → FFmpeg reads it directly (no local video)
      - Direct link  → FFmpeg reads URL directly (no local video)

    Args:
        url:         Source URL (Google Drive, Loom, or direct link)
        dest_dir:    Directory to write the MP3 into
        progress_cb: Optional callable(message: str) for progress updates

    Returns:
        Path to the .mp3 file in dest_dir
    """
    if not shutil.which("ffmpeg"):
        raise ValueError("FFmpeg is not installed. Install with: brew install ffmpeg")

    dest_dir.mkdir(parents=True, exist_ok=True)
    mp3_path = dest_dir / "recording.mp3"
    source   = detect_source(url)

    def _log(msg: str):
        logger.info(msg)
        if progress_cb:
            progress_cb(msg)

    # ── Google Drive ──────────────────────────────────────────────────────
    if source == "gdrive":
        try:
            import gdown
        except ImportError:
            raise ValueError(
                "gdown is required for Google Drive downloads. "
                "Install with: pip install gdown"
            )

        file_id   = _gdrive_id(url)
        video_tmp = dest_dir / "recording_video.mp4"

        _log(f"Downloading from Google Drive ({file_id[:12]}…) — this may take a few minutes…")
        try:
            gdown.download(
                id=file_id,
                output=str(video_tmp),
                quiet=False,
            )
        except Exception as e:
            raise ValueError(
                f"Google Drive download failed: {e}. "
                "Ensure the file is shared as 'Anyone with the link can view'."
            )

        if not video_tmp.exists() or video_tmp.stat().st_size < 10_000:
            video_tmp.unlink(missing_ok=True)
            raise ValueError(
                "Google Drive returned an empty or invalid file. "
                "Check the sharing setting: 'Anyone with the link can view'."
            )

        size_mb = video_tmp.stat().st_size / 1024 / 1024
        _log(f"Download complete: {size_mb:.0f}MB video — extracting audio…")

        try:
            _ffmpeg_extract_mp3(str(video_tmp), mp3_path)
        finally:
            # Always delete the raw video immediately after extraction
            video_tmp.unlink(missing_ok=True)
            _log("Video file deleted (only MP3 retained)")

        mp3_mb = mp3_path.stat().st_size / 1024 / 1024
        _log(f"Audio ready: {mp3_mb:.1f}MB MP3")
        return mp3_path

    # ── Loom ──────────────────────────────────────────────────────────────
    if source == "loom":
        video_id = _loom_id(url)
        _log("Resolving Loom CDN URL…")
        cdn_url  = _loom_direct_url(video_id)
        _log("Extracting audio from Loom (streaming via FFmpeg)…")
        _ffmpeg_extract_mp3(cdn_url, mp3_path)
        mp3_mb = mp3_path.stat().st_size / 1024 / 1024
        _log(f"Audio ready: {mp3_mb:.1f}MB MP3")
        return mp3_path

    # ── Direct URL ────────────────────────────────────────────────────────
    _log(f"Extracting audio from URL (streaming via FFmpeg)…")
    _ffmpeg_extract_mp3(url.strip(), mp3_path)
    mp3_mb = mp3_path.stat().st_size / 1024 / 1024
    _log(f"Audio ready: {mp3_mb:.1f}MB MP3")
    return mp3_path
