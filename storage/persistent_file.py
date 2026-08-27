"""
storage/persistent_file.py
Thin helper for "a single local file that must survive a Render restart",
used by the handful of modules that manage one fixed file rather than a
collection of per-record objects (reports/audit_excel_manager.py's master
tracker, reports/google_sheets_manager.py's OAuth files,
utils/word_normalizer.py, reports/tl_mapping.py).

Pattern: each such module keeps its existing local Path constant and all
of its existing read/write logic completely unchanged. Only two calls are
added at existing boundaries:
  - sync_from_r2_if_missing(path, key) right before the module would check
    `path.exists()` / read it — recovers the file from R2 first if this is
    a fresh instance (or the file was never written locally yet).
  - sync_to_r2(path, key) right after the module writes/saves the file —
    persists the new version to R2.

Both are no-ops when STORAGE_BACKEND=local (the default), so none of the
wrapped modules' behavior changes at all unless R2 is explicitly enabled.
Errors from R2 are logged, never silently swallowed, and never raised in
a way that blocks the *local* read/write the caller is also doing — a
transient R2 problem degrades to "local-only for this operation", it
doesn't crash the request. (This mirrors how the app already tolerates a
slow/unavailable Sarvam/OpenAI call elsewhere: log loudly, don't take the
whole request down for a non-critical persistence step.)
"""
from __future__ import annotations
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def sync_from_r2_if_missing(local_path: Path, key: str) -> None:
    """Keeps `local_path` and R2's `key` reconciled in the two directions
    that matter for this app's singleton config files. No-op in local
    mode either way.

      - local_path MISSING (fresh/restarted instance): download it from
        R2 first. No-op (logged, not raised) if R2 doesn't have it yet
        either — that's the normal "first time ever" case, the caller's
        existing create-if-missing logic takes over from here.
      - local_path EXISTS but R2 doesn't have `key` yet: upload it —
        this is the one-time backfill that matters the first time R2 is
        switched on for a file that was already sitting on local disk
        (e.g. a Google OAuth token established before this feature
        existed). Without this, that file would never reach R2 until it
        happened to be rewritten, and would be silently lost on the very
        next restart despite "looking" persisted.
      - local_path exists AND R2 already has `key`: no-op — local wins,
        never overwrites a possibly newer R2 copy just from a read.
    """
    from storage.backend import is_r2_enabled
    if not is_r2_enabled():
        return
    from storage.r2 import (
        download_file, upload_file, object_exists, StorageNotFoundError, StorageError,
    )
    if local_path.exists():
        try:
            if not object_exists(key):
                upload_file(local_path, key)
                logger.info(f"Backfilled {local_path.name} to R2 (key={key}) — "
                            f"existed locally but not in R2 yet")
        except StorageError as exc:
            logger.warning(f"Could not backfill {local_path.name} to R2 ({type(exc).__name__}): {exc}")
        return
    try:
        download_file(key, local_path)
        logger.info(f"Recovered {local_path.name} from R2 (key={key})")
    except StorageNotFoundError:
        pass  # nothing in R2 yet — fine, caller will create it fresh
    except StorageError as exc:
        logger.warning(f"Could not recover {local_path.name} from R2 ({type(exc).__name__}): {exc}")


def sync_to_r2(local_path: Path, key: str, content_type: str | None = None) -> None:
    """If R2 is enabled and `local_path` exists, upload its current bytes
    to R2. No-op in local mode. Logs (never silently swallows) on
    failure — the local write the caller just did still succeeds either
    way, but you'll see it in the logs if R2 persistence is failing."""
    from storage.backend import is_r2_enabled
    if not is_r2_enabled():
        return
    if not local_path.exists():
        logger.warning(f"sync_to_r2: {local_path} does not exist locally — nothing to upload")
        return
    from storage.r2 import upload_file, StorageError
    try:
        upload_file(local_path, key, content_type=content_type)
    except StorageError as exc:
        logger.error(f"Failed to persist {local_path.name} to R2 (key={key}): {type(exc).__name__}: {exc}")


def delete_from_r2(key: str) -> None:
    """If R2 is enabled, delete `key`. No-op in local mode. Logs on
    failure rather than raising, matching sync_to_r2's tolerance."""
    from storage.backend import is_r2_enabled
    if not is_r2_enabled():
        return
    from storage.r2 import delete_file, StorageError
    try:
        delete_file(key)
    except StorageError as exc:
        logger.error(f"Failed to delete R2 object (key={key}): {type(exc).__name__}: {exc}")
