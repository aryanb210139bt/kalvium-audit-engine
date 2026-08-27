"""
storage/backend.py
Central switch for which object-storage backend the app uses — mirrors
db/backend.py's pattern exactly, but is entirely independent of it.

Defaults to "local" (current behavior, zero risk). Only set
STORAGE_BACKEND=r2 once R2_ENDPOINT_URL/R2_BUCKET_NAME/R2_ACCESS_KEY_ID/
R2_SECRET_ACCESS_KEY are configured.

SAFETY: r2_access_key_id / r2_secret_access_key must never be logged,
printed, or included in any exception message anywhere in this codebase.
"""
from __future__ import annotations
from config.settings import get_settings


def storage_backend() -> str:
    return (get_settings().storage_backend or "local").strip().lower()


def is_r2_enabled() -> bool:
    return storage_backend() == "r2"


def get_r2_config() -> dict:
    """Returns the R2 connection config for opening a client. Callers must
    never log, print, or surface r2_access_key_id/r2_secret_access_key —
    only use them to build a boto3 client."""
    s = get_settings()
    return {
        "endpoint_url": s.r2_endpoint_url or "",
        "bucket_name": s.r2_bucket_name or "kalvium-audit-storage",
        "access_key_id": s.r2_access_key_id or "",
        "secret_access_key": s.r2_secret_access_key or "",
        "region": s.r2_region or "auto",
    }


def r2_configured() -> bool:
    """Safe to log/print/return — reveals only whether values are set, not
    their contents."""
    cfg = get_r2_config()
    return bool(cfg["endpoint_url"] and cfg["access_key_id"] and cfg["secret_access_key"])
