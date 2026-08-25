"""
reports/google_sheets_manager.py
Handles Google OAuth2 flow and Sheets API writes.

Files used (all under data/):
  google_credentials.json  — OAuth client credentials (user uploads from Google Cloud Console)
  google_token.json        — Access + refresh token (auto-created after first auth)
  google_config.json       — Sheet ID + tab name chosen by the user
"""
from __future__ import annotations
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR          = Path("data")
CREDENTIALS_PATH  = DATA_DIR / "google_credentials.json"
TOKEN_PATH        = DATA_DIR / "google_token.json"
CONFIG_PATH       = DATA_DIR / "google_config.json"
_PKCE_VERIFIER_PATH = DATA_DIR / "google_oauth_verifier.txt"

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    # Read-only file listing (id/name only, not content) — needed so the app
    # can show you a picker of your spreadsheets in list_spreadsheets().
    "https://www.googleapis.com/auth/drive.metadata.readonly",
]


# ── Config helpers ────────────────────────────────────────────────────────────

def get_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text())
        except Exception:
            pass
    return {}


def save_config(cfg: dict):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


def has_credentials() -> bool:
    return CREDENTIALS_PATH.exists()


def has_token() -> bool:
    return TOKEN_PATH.exists()


# ── OAuth flow ────────────────────────────────────────────────────────────────

def get_auth_url(redirect_uri: str) -> str:
    """Build the Google OAuth consent URL and return it."""
    from google_auth_oauthlib.flow import Flow
    flow = Flow.from_client_secrets_file(str(CREDENTIALS_PATH), scopes=SCOPES)
    flow.redirect_uri = redirect_uri
    auth_url, _ = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )
    # Flow auto-generates a PKCE code_verifier per instance. The callback
    # request builds a *separate* Flow to exchange the code, so we have to
    # persist this one's verifier to disk (single-user, single-flow-at-a-time
    # local app — a file is enough, no session store needed) or the token
    # exchange fails with "invalid_grant: Missing code verifier".
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _PKCE_VERIFIER_PATH.write_text(flow.code_verifier)
    return auth_url


def exchange_code(code: str, redirect_uri: str) -> dict:
    """Exchange auth code for tokens and persist them. Returns user info."""
    from google_auth_oauthlib.flow import Flow
    from googleapiclient.discovery import build

    code_verifier = None
    if _PKCE_VERIFIER_PATH.exists():
        code_verifier = _PKCE_VERIFIER_PATH.read_text().strip()
        _PKCE_VERIFIER_PATH.unlink()   # one-time use

    flow = Flow.from_client_secrets_file(
        str(CREDENTIALS_PATH), scopes=SCOPES, code_verifier=code_verifier
    )
    flow.redirect_uri = redirect_uri
    flow.fetch_token(code=code)

    creds = flow.credentials
    _save_token(creds)

    # Get user email via tokeninfo
    try:
        import urllib.request, urllib.parse
        info_url = f"https://www.googleapis.com/oauth2/v3/tokeninfo?access_token={creds.token}"
        with urllib.request.urlopen(info_url) as resp:
            info = json.loads(resp.read())
        return {"email": info.get("email", ""), "ok": True}
    except Exception:
        return {"email": "", "ok": True}


def _save_token(creds):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TOKEN_PATH.write_text(creds.to_json())


def get_credentials():
    """Return valid (auto-refreshed) credentials or None."""
    if not TOKEN_PATH.exists():
        return None
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            _save_token(creds)
        except Exception as e:
            logger.warning(f"Token refresh failed: {e}")
            return None
    return creds if (creds and creds.valid) else None


def get_connected_email() -> str:
    """Return the email stored in the token file, or empty string."""
    if not TOKEN_PATH.exists():
        return ""
    try:
        data = json.loads(TOKEN_PATH.read_text())
        # token info endpoint
        import urllib.request
        access = data.get("token", "")
        if not access:
            return ""
        url = f"https://www.googleapis.com/oauth2/v3/tokeninfo?access_token={access}"
        with urllib.request.urlopen(url, timeout=4) as resp:
            info = json.loads(resp.read())
        return info.get("email", "")
    except Exception:
        return ""


def disconnect():
    """Remove stored token and config."""
    if TOKEN_PATH.exists():
        TOKEN_PATH.unlink()


# ── Sheet operations ──────────────────────────────────────────────────────────

def list_spreadsheets() -> list[dict]:
    """List the user's Google Sheets files (name + id)."""
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
        raise RuntimeError("Not authenticated with Google")

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
    from reports.google_sheets_manager import get_credentials

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
