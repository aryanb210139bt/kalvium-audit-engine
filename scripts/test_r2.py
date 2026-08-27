"""
scripts/test_r2.py
Manual, read-and-write connectivity diagnostic for Cloudflare R2. Uploads
one tiny, harmless test object, verifies it exists, downloads it, checks
the contents match, then deletes it — reporting PASS/FAIL at each step.

Usage:
    PYTHONPATH="$(pwd)" python3 scripts/test_r2.py

Requires STORAGE_BACKEND=r2 and R2_ENDPOINT_URL/R2_BUCKET_NAME/
R2_ACCESS_KEY_ID/R2_SECRET_ACCESS_KEY to be set (in .env or the
environment). Never prints any credential — only a fixed, generic message
on failure.
"""
from __future__ import annotations
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TEST_KEY_PREFIX = "_connectivity_test/"


def _redact(exc: Exception) -> str:
    return type(exc).__name__


def run() -> int:
    from storage.backend import r2_configured, get_r2_config

    if not r2_configured():
        print("FAIL: R2 is not configured. Set R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, "
              "and R2_SECRET_ACCESS_KEY (in .env or the environment) and try again.")
        return 1

    cfg = get_r2_config()
    print(f"Bucket: {cfg['bucket_name']}  |  Region: {cfg['region']}")
    print("(endpoint URL and credentials are configured but not printed)")

    from storage import r2 as storage_r2

    key = f"{TEST_KEY_PREFIX}{uuid.uuid4().hex}.txt"
    payload = f"kalvium-audit-engine R2 connectivity test — {time.time()}".encode()

    steps = []

    def step(name: str, fn) -> bool:
        try:
            fn()
            print(f"  [PASS] {name}")
            steps.append(True)
            return True
        except Exception as exc:
            print(f"  [FAIL] {name} — {type(exc).__name__}: {_redact_message(exc)}")
            steps.append(False)
            return False

    def _redact_message(exc: Exception) -> str:
        # storage/r2.py's own exceptions already carry a generic message
        # with no credentials — safe to print directly.
        return str(exc)

    print("\nRunning R2 connectivity test...")

    ok = step("Connect + upload test object", lambda: storage_r2.upload_bytes(payload, key, content_type="text/plain"))
    if not ok:
        return _finish(steps)

    step("Verify object exists", lambda: _assert(storage_r2.object_exists(key), "object_exists() returned False"))

    downloaded = {}

    def _download():
        downloaded["data"] = storage_r2.download_bytes(key)

    if step("Download test object", _download):
        step("Verify downloaded contents match", lambda: _assert(
            downloaded.get("data") == payload, "downloaded bytes did not match what was uploaded"
        ))

    step("Delete test object", lambda: storage_r2.delete_file(key))
    step("Verify object no longer exists", lambda: _assert(
        not storage_r2.object_exists(key), "object still exists after delete"
    ))

    return _finish(steps)


def _assert(cond: bool, message: str) -> None:
    if not cond:
        raise AssertionError(message)


def _finish(steps: list[bool]) -> int:
    print()
    if steps and all(steps):
        print("R2 CONNECTIVITY: PASS")
        return 0
    print("R2 CONNECTIVITY: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(run())
