"""
storage/r2.py
Cloudflare R2 object storage client — S3-compatible, via boto3. Used only
when STORAGE_BACKEND=r2 (see storage/backend.py). This is the ONLY place
in the codebase that talks to boto3/R2 directly; every other module goes
through the functions here (or storage/persistent_file.py's thin wrapper
for "one file synced with a local path") rather than scattering boto3
calls throughout the application.

R2 configuration (see config/settings.py / .env.example):
    R2_ENDPOINT_URL=https://<account-id>.r2.cloudflarestorage.com
    R2_BUCKET_NAME=kalvium-audit-storage
    R2_ACCESS_KEY_ID=...
    R2_SECRET_ACCESS_KEY=...
    R2_REGION=auto

The bucket is treated as PRIVATE throughout — nothing here makes an
object public. Callers that need to hand a file to a browser/user should
use generate_presigned_download_url(), not a raw object URL.

SAFETY: r2_access_key_id/r2_secret_access_key are used only to construct
the boto3 client. They are never logged, printed, or included in any
exception message — every error path below raises a fixed, generic
message and logs only the exception's class name.
"""
from __future__ import annotations
import logging
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import boto3
    from botocore.config import Config as BotoConfig
    from botocore.exceptions import ClientError, EndpointConnectionError, NoCredentialsError
except ImportError:  # only a problem if STORAGE_BACKEND=r2 is actually selected
    boto3 = None
    BotoConfig = None
    ClientError = EndpointConnectionError = NoCredentialsError = Exception


# ── Errors ───────────────────────────────────────────────────────────────────

class StorageError(RuntimeError):
    """Base class for all storage/r2 errors. Message is always generic —
    never includes credentials."""


class StorageConfigError(StorageError):
    """R2 selected but not configured, or the boto3/dependency isn't
    installed."""


class StorageUnavailableError(StorageError):
    """Could not reach R2 (network/DNS/connection issue)."""


class StorageNotFoundError(StorageError):
    """The requested object does not exist in the bucket."""


class StorageUploadError(StorageError):
    """An upload (file or bytes) failed."""


class StorageDownloadError(StorageError):
    """A download (file or bytes) failed for a reason other than
    not-found."""


class StorageDeleteError(StorageError):
    """A delete failed."""


def _redact(exc: Exception) -> str:
    """Never let a raw boto3/botocore exception (which can, in some error
    paths, echo back request details) reach a log line unfiltered beyond
    its class name."""
    return type(exc).__name__


# ── Client ───────────────────────────────────────────────────────────────────

_local = threading.local()


def get_r2_client():
    """Returns a boto3 S3 client configured for R2, one per thread. Raises
    StorageConfigError (generic message, no credentials) if R2 isn't
    configured or boto3 isn't installed."""
    from storage.backend import get_r2_config

    if boto3 is None:
        raise StorageConfigError(
            "STORAGE_BACKEND=r2 but the 'boto3' package is not installed. "
            "Run: pip install boto3"
        )
    cfg = get_r2_config()
    if not (cfg["endpoint_url"] and cfg["access_key_id"] and cfg["secret_access_key"]):
        raise StorageConfigError(
            "STORAGE_BACKEND=r2 but R2 is not fully configured — check "
            "R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, and R2_SECRET_ACCESS_KEY."
        )
    if not hasattr(_local, "client") or _local.client is None:
        try:
            _local.client = boto3.client(
                "s3",
                endpoint_url=cfg["endpoint_url"],
                aws_access_key_id=cfg["access_key_id"],
                aws_secret_access_key=cfg["secret_access_key"],
                region_name=cfg["region"] or "auto",
                config=BotoConfig(signature_version="s3v4"),
            )
        except Exception:
            raise StorageConfigError(
                "Could not create the R2 client. Check R2_ENDPOINT_URL and credentials."
            ) from None
    return _local.client


def _bucket_name() -> str:
    from storage.backend import get_r2_config
    return get_r2_config()["bucket_name"]


# ── Core operations ──────────────────────────────────────────────────────────

def upload_file(local_path: Path | str, key: str, content_type: Optional[str] = None) -> None:
    """Uploads a local file to R2 under `key`."""
    local_path = Path(local_path)
    if not local_path.exists():
        raise StorageUploadError(f"Local file not found for upload: {local_path.name}")
    client = get_r2_client()
    extra = {"ContentType": content_type} if content_type else {}
    try:
        client.upload_file(str(local_path), _bucket_name(), key, ExtraArgs=extra or None)
    except FileNotFoundError:
        raise StorageUploadError(f"Local file not found for upload: {local_path.name}") from None
    except (EndpointConnectionError,) as exc:
        raise StorageUnavailableError(f"R2 unavailable during upload ({_redact(exc)})") from None
    except (ClientError, NoCredentialsError) as exc:
        raise StorageUploadError(f"R2 upload failed ({_redact(exc)})") from None
    except Exception as exc:
        raise StorageUploadError(f"R2 upload failed ({_redact(exc)})") from None


def upload_bytes(data: bytes, key: str, content_type: Optional[str] = None) -> None:
    """Uploads an in-memory byte string to R2 under `key`."""
    client = get_r2_client()
    kwargs = {"Bucket": _bucket_name(), "Key": key, "Body": data}
    if content_type:
        kwargs["ContentType"] = content_type
    try:
        client.put_object(**kwargs)
    except (EndpointConnectionError,) as exc:
        raise StorageUnavailableError(f"R2 unavailable during upload ({_redact(exc)})") from None
    except (ClientError, NoCredentialsError) as exc:
        raise StorageUploadError(f"R2 upload failed ({_redact(exc)})") from None
    except Exception as exc:
        raise StorageUploadError(f"R2 upload failed ({_redact(exc)})") from None


def download_file(key: str, local_path: Path | str) -> None:
    """Downloads an R2 object to a local file path."""
    client = get_r2_client()
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        client.download_file(_bucket_name(), key, str(local_path))
    except (EndpointConnectionError,) as exc:
        raise StorageUnavailableError(f"R2 unavailable during download ({_redact(exc)})") from None
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchKey"):
            raise StorageNotFoundError(f"Object not found: {key}") from None
        raise StorageDownloadError(f"R2 download failed ({_redact(exc)})") from None
    except Exception as exc:
        raise StorageDownloadError(f"R2 download failed ({_redact(exc)})") from None


def download_bytes(key: str) -> bytes:
    """Downloads an R2 object and returns its bytes."""
    client = get_r2_client()
    try:
        resp = client.get_object(Bucket=_bucket_name(), Key=key)
        return resp["Body"].read()
    except (EndpointConnectionError,) as exc:
        raise StorageUnavailableError(f"R2 unavailable during download ({_redact(exc)})") from None
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchKey"):
            raise StorageNotFoundError(f"Object not found: {key}") from None
        raise StorageDownloadError(f"R2 download failed ({_redact(exc)})") from None
    except Exception as exc:
        raise StorageDownloadError(f"R2 download failed ({_redact(exc)})") from None


def delete_file(key: str) -> None:
    """Deletes an object from R2. Not an error if it doesn't exist (matches
    S3/R2's own delete_object semantics — idempotent)."""
    client = get_r2_client()
    try:
        client.delete_object(Bucket=_bucket_name(), Key=key)
    except (EndpointConnectionError,) as exc:
        raise StorageUnavailableError(f"R2 unavailable during delete ({_redact(exc)})") from None
    except (ClientError, NoCredentialsError) as exc:
        raise StorageDeleteError(f"R2 delete failed ({_redact(exc)})") from None
    except Exception as exc:
        raise StorageDeleteError(f"R2 delete failed ({_redact(exc)})") from None


def object_exists(key: str) -> bool:
    """Returns True/False — never raises for a simple not-found; only
    raises StorageUnavailableError/StorageConfigError for a genuine
    connectivity/config problem, so callers can distinguish "definitely
    doesn't exist" from "couldn't check"."""
    client = get_r2_client()
    try:
        client.head_object(Bucket=_bucket_name(), Key=key)
        return True
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        raise StorageUnavailableError(f"Could not check object existence ({_redact(exc)})") from None
    except (EndpointConnectionError,) as exc:
        raise StorageUnavailableError(f"R2 unavailable ({_redact(exc)})") from None


def get_object_url(key: str) -> str:
    """Returns an internal reference string for this object — NOT a
    fetchable public URL (the bucket is private). Safe to store in a
    database as a record of where the object lives. To actually hand a
    file to a user/browser, use generate_presigned_download_url()."""
    return f"r2://{_bucket_name()}/{key}"


def generate_presigned_download_url(key: str, expires_in: int = 900) -> str:
    """Returns a short-lived, signed GET URL (default 15 minutes) a
    browser can fetch directly — the bucket itself stays private.
    `expires_in` is in seconds; 900-3600 (15-60 min) is the recommended
    range for this app's flows."""
    client = get_r2_client()
    try:
        return client.generate_presigned_url(
            "get_object",
            Params={"Bucket": _bucket_name(), "Key": key},
            ExpiresIn=expires_in,
        )
    except Exception as exc:
        raise StorageDownloadError(f"Could not generate presigned download URL ({_redact(exc)})") from None


def generate_presigned_upload_url(key: str, expires_in: int = 900, content_type: Optional[str] = None) -> str:
    """Returns a short-lived, signed PUT URL a client could upload directly
    to (useful for browser-direct uploads later, if ever needed — not used
    by the server-side upload flows today)."""
    client = get_r2_client()
    params = {"Bucket": _bucket_name(), "Key": key}
    if content_type:
        params["ContentType"] = content_type
    try:
        return client.generate_presigned_url("put_object", Params=params, ExpiresIn=expires_in)
    except Exception as exc:
        raise StorageUploadError(f"Could not generate presigned upload URL ({_redact(exc)})") from None


def health_check() -> tuple[bool, str]:
    """Returns (ok, message). `message` is always a short constant string —
    safe to log or return from an API endpoint — never credentials and
    never a raw driver exception message. Checks bucket reachability via
    head_bucket (no object required)."""
    from storage.backend import r2_configured
    if not r2_configured():
        return False, "R2 is not configured"
    if boto3 is None:
        return False, "boto3 package is not installed"
    try:
        client = get_r2_client()
        client.head_bucket(Bucket=_bucket_name())
        return True, "R2 connection OK"
    except Exception:
        return False, "R2 connection failed"
