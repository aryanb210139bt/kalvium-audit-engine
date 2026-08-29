"""
reports/google_sheets_manager.py
Google Sheets/Drive integration via a SERVICE ACCOUNT — not OAuth2 user
consent.

Why the switch: OAuth2 needed a human to click through a consent screen
and its access token needed periodic refresh (and could silently expire/
revoke). A service account is its own Google identity — no interactive
consent, no per-user token to refresh or lose. You share the target
spreadsheet (or Drive folder) with the service account's own email
address once, the same way you'd share a doc with a colleague, and it
keeps working across every restart with zero re-auth step.

File used (under data/, persisted to R2 — see storage/persistent_file.py):
  google_service_account.json — the service account key downloaded from
  Google Cloud Console (IAM & Admin -> Service Accounts -> Keys -> Add
  key -> JSON). Contains a private key — never log, print, or commit it.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR              = Path("data")
SERVICE_ACCOUNT_PATH   = DATA_DIR / "google_service_account.json"
CONFIG_PATH            = DATA_DIR / "google_config.json"

# R2 keys — see storage/persistent_file.py. No-ops unless STORAGE_BACKEND=r2.
_R2_KEY_SERVICE_ACCOUNT = "integrations/google_sheets/service_account.json"
_R2_KEY_CONFIG          = "integrations/google_sheets/config.json"

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    # Read-only file listing (id/name only, not content) — needed so the app
    # can show a picker of spreadsheets the service account can see in
    # list_spreadsheets().
    "https://www.googleapis.com/auth/drive.metadata.readonly",
]


# ── Config helpers ────────────────────────────────────────────────────────────

def get_config() -> dict:
    from storage.persistent_file import sync_from_r2_if_missing
    sync_from_r2_if_missing(CONFIG_PATH, _R2_KEY_CONFIG)
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text())
        except Exception:
            pass
    return {}


def save_config(cfg: dict):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    from storage.persistent_file import sync_to_r2
    sync_to_r2(CONFIG_PATH, _R2_KEY_CONFIG, content_type="application/json")


# ── Service account credential management ───────────────────────────────────

def has_service_account() -> bool:
    from storage.persistent_file import sync_from_r2_if_missing
    sync_from_r2_if_missing(SERVICE_ACCOUNT_PATH, _R2_KEY_SERVICE_ACCOUNT)
    return SERVICE_ACCOUNT_PATH.exists()


def save_service_account(content: bytes) -> None:
    """Persist an uploaded service-account JSON key (validated by the
    caller before this is invoked — see api/main.py's upload endpoint)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SERVICE_ACCOUNT_PATH.write_bytes(content)
    from storage.persistent_file import sync_to_r2
    sync_to_r2(SERVICE_ACCOUNT_PATH, _R2_KEY_SERVICE_ACCOUNT, content_type="application/json")


def get_service_account_email() -> str:
    """Returns the service account's own email (client_email from the key
    file) — this is the address you share a Sheet/Drive file with in
    Google's sharing dialog. Empty string if not configured or unreadable."""
    if not has_service_account():
        return ""
    try:
        data = json.loads(SERVICE_ACCOUNT_PATH.read_text())
        return data.get("client_email", "")
    except Exception:
        return ""


def get_credentials():
    """Return service-account credentials, or None if not configured. No
    refresh/expiry handling needed here — the google-auth library renews
    the short-lived access token from the private key transparently on
    each API call."""
    if not has_service_account():
        return None
    from google.oauth2.service_account import Credentials
    try:
        return Credentials.from_service_account_file(str(SERVICE_ACCOUNT_PATH), scopes=SCOPES)
    except Exception as e:
        logger.warning(f"Could not load service account credentials: {e}")
        return None


def disconnect():
    """Remove the stored service account key, locally and from R2."""
    from storage.persistent_file import delete_from_r2
    if SERVICE_ACCOUNT_PATH.exists():
        SERVICE_ACCOUNT_PATH.unlink()
    delete_from_r2(_R2_KEY_SERVICE_ACCOUNT)


# ── Sheet operations (unchanged — all just consume get_credentials()) ──────────

def list_spreadsheets() -> list[dict]:
    """List spreadsheets the service account can see (name + id) — only
    ones explicitly shared with its email address."""
    creds = get_credentials()
    if not creds:
        return []
    from googleapiclient.discovery import build
    try:
        drive = build("drive", "v3", credentials=creds)
        result = drive.files().list(
            q="mimeType='application/vnd.google-apps.spreadsheet'",
            fields="files(id,name)",
            pageSize=50,
            orderBy="modifiedTime desc",
        ).execute()
        return result.get("files", [])
    except Exception as e:
        logger.error(f"list_spreadsheets error: {e}")
        return []


def get_sheet_tabs(sheet_id: str) -> list[str]:
    """Return all sheet tab names for the given spreadsheet."""
    creds = get_credentials()
    if not creds:
        return []
    from googleapiclient.discovery import build
    try:
        service = build("sheets", "v4", credentials=creds)
        meta = service.spreadsheets().get(spreadsheetId=sheet_id).execute()
        return [s["properties"]["title"] for s in meta.get("sheets", [])]
    except Exception as e:
        logger.error(f"get_sheet_tabs error: {e}")
        return []


def _col_letter(idx0: int) -> str:
    """0-indexed column number -> spreadsheet column letters (0->A, 25->Z, 26->AA, ...)."""
    idx1 = idx0 + 1
    letters = ""
    while idx1 > 0:
        idx1, rem = divmod(idx1 - 1, 26)
        letters = chr(65 + rem) + letters
    return letters


def append_row_to_sheet(row_data: dict) -> dict:
    """
    Write one row to the configured Google Sheet. If this row's "Session ID"
    already exists in the sheet (this audit was pushed before and is being
    edited/re-pushed), that existing row is updated in place instead of
    appending a duplicate.
    Returns {"updated_range": "...", "updated_rows": n}.
    """
    from reports.audit_excel_manager import COLUMNS
    from googleapiclient.discovery import build

    creds = get_credentials()
    if not creds:
        raise RuntimeError("Google Sheets is not configured — upload a service account key first")

    cfg = get_config()
    sheet_id = cfg.get("sheet_id", "")
    tab_name = cfg.get("tab_name", "2026 Demo Auditng")
    if not sheet_id:
        raise RuntimeError("No Google Sheet configured")

    # Build row in exact COLUMNS order
    values = [[str(row_data.get(col, "") or "") for col in COLUMNS]]
    service = build("sheets", "v4", credentials=creds)

    target_sid = str(row_data.get("Session ID", "")).strip()
    existing_row_number = None
    if target_sid:
        sid_col = _col_letter(COLUMNS.index("Session ID"))
        resp = service.spreadsheets().values().get(
            spreadsheetId=sheet_id, range=f"'{tab_name}'!{sid_col}2:{sid_col}"
        ).execute()
        for i, row in enumerate(resp.get("values", [])):
            if row and str(row[0]).strip() == target_sid:
                existing_row_number = i + 2   # +2: header row + 0-index offset
                break

    if existing_row_number:
        result = service.spreadsheets().values().update(
            spreadsheetId=sheet_id,
            range=f"'{tab_name}'!A{existing_row_number}",
            valueInputOption="USER_ENTERED",
            body={"values": values},
        ).execute()
        return {"updated_range": result.get("updatedRange", ""), "updated_rows": 1}

    result = service.spreadsheets().values().append(
        spreadsheetId=sheet_id,
        range=f"'{tab_name}'!A1",
        valueInputOption="USER_ENTERED",
        insertDataOption="INSERT_ROWS",
        body={"values": values},
    ).execute()
    updates = result.get("updates", {})
    return {
        "updated_range": updates.get("updatedRange", ""),
        "updated_rows": updates.get("updatedRows", 0),
    }


def ensure_header_row(sheet_id: str, tab_name: str):
    """Write the header row if row 1 is empty."""
    from reports.audit_excel_manager import COLUMNS
    from googleapiclient.discovery import build

    creds = get_credentials()
    if not creds:
        return

    service = build("sheets", "v4", credentials=creds)
    # Read row 1
    try:
        resp = service.spreadsheets().values().get(
            spreadsheetId=sheet_id,
            range=f"'{tab_name}'!1:1",
        ).execute()
        existing = resp.get("values", [[]])
        if existing and existing[0]:
            return  # already has headers
    except Exception:
        pass

    # Write headers
    service.spreadsheets().values().update(
        spreadsheetId=sheet_id,
        range=f"'{tab_name}'!A1",
        valueInputOption="RAW",
        body={"values": [list(COLUMNS)]},
    ).execute()
    logger.info(f"Wrote headers to {sheet_id} / {tab_name}")
