"""
utils/sheets_writer.py
Google Sheets output integration for Kalvium Audit Platform.

Writes audit results to 4 sheets:
  1. AI_Audit_Results   — full per-call audit log
  2. Associate_Analytics — rolling associate-level stats
  3. Fraud_Analytics     — fraud flags and pattern tracking
  4. Attendance_Tracking — attendance prediction vs actuals

Requires: google-auth, google-auth-oauthlib, google-api-python-client
Install:   pip install google-auth google-auth-httplib2 google-api-python-client

Configuration (add to .env):
  GOOGLE_SHEETS_CREDENTIALS_PATH=credentials.json
  GOOGLE_SHEETS_SPREADSHEET_ID=<your-spreadsheet-id>
"""
from __future__ import annotations
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from config.kalvium_models import KalviumAuditReport

logger = logging.getLogger(__name__)

_CREDENTIALS_PATH   = os.getenv("GOOGLE_SHEETS_CREDENTIALS_PATH", "credentials.json")
_SPREADSHEET_ID     = os.getenv("GOOGLE_SHEETS_SPREADSHEET_ID", "")
_SCOPES             = ["https://www.googleapis.com/auth/spreadsheets"]


def _get_service():
    """Build and return an authenticated Google Sheets service object."""
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build

        creds = service_account.Credentials.from_service_account_file(
            _CREDENTIALS_PATH, scopes=_SCOPES
        )
        return build("sheets", "v4", credentials=creds, cache_discovery=False)
    except Exception as exc:
        logger.error(f"Google Sheets auth failed: {exc}")
        return None


def _append_rows(service, sheet_name: str, rows: list[list[Any]]) -> bool:
    """Append rows to a named sheet tab."""
    if not _SPREADSHEET_ID:
        logger.warning("GOOGLE_SHEETS_SPREADSHEET_ID not set — skipping write")
        return False
    try:
        body = {"values": rows}
        service.spreadsheets().values().append(
            spreadsheetId=_SPREADSHEET_ID,
            range=f"{sheet_name}!A1",
            valueInputOption="RAW",
            insertDataOption="INSERT_ROWS",
            body=body,
        ).execute()
        return True
    except Exception as exc:
        logger.error(f"Sheets append to '{sheet_name}' failed: {exc}")
        return False


def _fmt(val: Any) -> str:
    """Flatten a value to a Sheets-safe string."""
    if isinstance(val, (list, dict)):
        return json.dumps(val, ensure_ascii=False)[:500]
    if val is None:
        return ""
    return str(val)


# ── Sheet header definitions ──────────────────────────────────────────────────

_AUDIT_HEADERS = [
    "audit_id", "associate_id", "associate_name", "lead_id", "prospect_name",
    "call_date", "duration_seconds", "language_detected",
    "talk_ratio_associate", "talk_ratio_prospect", "talk_ratio_parent", "silence_ratio",
    # Layer 1
    "compliance_score", "qualification_completed", "email_collected", "email_verified",
    "webinar_pitched", "attendance_commitment_obtained", "missing_steps",
    # Layer 2
    "interest_score", "buying_intent_score", "question_depth_score",
    "career_seriousness", "parent_involved", "fee_discussion", "placement_interest",
    "questions_asked_by_prospect",
    # Layer 3
    "engagement_score", "attention_level", "emotional_engagement",
    "passive_responses_count", "one_word_replies_count", "associate_dominated",
    # Layer 4
    "booking_authenticity_score", "fake_booking_probability", "fraud_risk_level",
    "tier1_flags", "tier2_flags", "red_flags", "relationship_indicators",
    "email_risk_level", "email_repeated_by_associate", "webinar_joining_discussed",
    "buying_journey_present", "natural_objections_present",
    # Layer 5
    "attendance_probability", "commitment_language_score", "time_confirmed",
    "joining_instructions_discussed",
    # Final
    "final_classification", "recommended_action", "urgency",
    "manager_summary", "top_concerns", "coaching_needs",
    "created_at",
]

_FRAUD_HEADERS = [
    "audit_id", "associate_id", "detection_date", "fraud_risk_level",
    "fake_booking_probability", "final_classification",
    "tier1_flags", "tier2_flags", "red_flags_count",
    "tier1_flag_details", "email_risk_level", "call_duration_seconds",
    "relationship_indicators", "coaching_indicators",
    "authenticity_reasoning", "recommended_action", "urgency",
]

_ATTENDANCE_HEADERS = [
    "audit_id", "associate_id", "lead_id", "call_date",
    "attendance_probability", "commitment_language_score",
    "joining_instructions_discussed", "time_confirmed",
    "follow_through_indicators", "risk_factors",
    "prediction_confidence", "prediction_reasoning",
    "actually_attended",   # filled later
    "no_show_reason",      # filled later
    "prediction_accuracy", # filled later
]


# ── Public write functions ────────────────────────────────────────────────────

def write_audit_result(report: KalviumAuditReport) -> bool:
    """Write a completed Kalvium audit report to Google Sheets."""
    service = _get_service()
    if not service:
        _write_local_fallback(report)
        return False

    c  = report.compliance
    bi = report.buyer_intent
    en = report.engagement
    au = report.authenticity
    at = report.attendance
    ms = report.manager_summary

    row = [
        report.audit_id, report.associate_id, report.associate_name,
        report.lead_id, report.prospect_name,
        report.call_date, round(report.duration_seconds, 1),
        report.language_detected,
        report.talk_ratio_associate, report.talk_ratio_prospect,
        report.talk_ratio_parent, report.silence_ratio,
        # Layer 1
        round(c.compliance_score, 1), c.qualification_completed,
        c.email_collected, c.email_verified,
        c.webinar_pitched, c.attendance_commitment_obtained,
        _fmt(c.missing_steps),
        # Layer 2
        round(bi.interest_score, 1), round(bi.buying_intent_score, 1),
        round(bi.question_depth_score, 1),
        bi.career_seriousness, bi.parent_involved,
        bi.fee_discussion, bi.placement_interest,
        _fmt(bi.questions_asked_by_prospect),
        # Layer 3
        round(en.engagement_score, 1), en.attention_level,
        en.emotional_engagement,
        en.passive_responses_count, en.one_word_replies_count,
        en.associate_dominated,
        # Layer 4
        round(au.booking_authenticity_score, 1),
        round(au.fake_booking_probability, 3),
        au.fraud_risk_level,
        au.tier1_flags, au.tier2_flags,
        _fmt(au.red_flags),
        _fmt(au.relationship_indicators),
        au.email_analysis.email_risk_level,
        au.email_analysis.email_repeated_by_associate,
        au.email_analysis.webinar_joining_discussed,
        au.buying_journey_present, au.natural_objections_present,
        # Layer 5
        round(at.attendance_probability, 3),
        at.commitment_language_score,
        at.time_confirmed, at.joining_instructions_discussed,
        # Final
        report.final_classification, report.recommended_action,
        report.urgency,
        ms.manager_summary if ms else "",
        _fmt(ms.top_concerns if ms else []),
        _fmt(ms.coaching_needs if ms else []),
        datetime.utcnow().isoformat(),
    ]

    ok = _append_rows(service, "AI_Audit_Results", [row])

    # Also write to Fraud_Analytics if flagged
    if au.fraud_risk_level in ("HIGH", "MEDIUM"):
        fraud_row = [
            report.audit_id, report.associate_id,
            datetime.utcnow().strftime("%Y-%m-%d"),
            au.fraud_risk_level,
            round(au.fake_booking_probability, 3),
            report.final_classification,
            au.tier1_flags, au.tier2_flags,
            len(au.red_flags),
            _fmt(au.red_flags[:3]),
            au.email_analysis.email_risk_level,
            round(report.duration_seconds, 1),
            _fmt(au.relationship_indicators),
            _fmt(au.coaching_indicators),
            au.authenticity_reasoning[:300],
            report.recommended_action, report.urgency,
        ]
        _append_rows(service, "Fraud_Analytics", [fraud_row])

    # Write attendance prediction row
    att_row = [
        report.audit_id, report.associate_id, report.lead_id, report.call_date,
        round(at.attendance_probability, 3),
        at.commitment_language_score,
        at.joining_instructions_discussed,
        at.time_confirmed,
        _fmt(at.follow_through_indicators),
        _fmt(at.risk_factors),
        at.prediction_confidence,
        at.prediction_reasoning[:200],
        "",  # actually_attended — filled post-webinar
        "",  # no_show_reason
        "",  # prediction_accuracy
    ]
    _append_rows(service, "Attendance_Tracking", [att_row])

    return ok


def ensure_sheet_headers(sheet_name: str) -> bool:
    """Create header rows in empty sheets (call once on startup)."""
    service = _get_service()
    if not service or not _SPREADSHEET_ID:
        return False

    headers_map = {
        "AI_Audit_Results":   _AUDIT_HEADERS,
        "Fraud_Analytics":    _FRAUD_HEADERS,
        "Attendance_Tracking": _ATTENDANCE_HEADERS,
    }
    headers = headers_map.get(sheet_name)
    if not headers:
        return False

    try:
        result = service.spreadsheets().values().get(
            spreadsheetId=_SPREADSHEET_ID,
            range=f"{sheet_name}!A1:A1",
        ).execute()
        if result.get("values"):
            return True  # already has headers
        return _append_rows(service, sheet_name, [headers])
    except Exception as exc:
        logger.error(f"ensure_sheet_headers failed for '{sheet_name}': {exc}")
        return False


def _write_local_fallback(report: KalviumAuditReport) -> None:
    """When Sheets is unavailable, save JSON locally for later sync."""
    fallback_dir = Path("audit_exports")
    fallback_dir.mkdir(exist_ok=True)
    out_path = fallback_dir / f"{report.audit_id}.json"
    out_path.write_text(report.model_dump_json(indent=2))
    logger.info(f"Sheets unavailable — saved locally: {out_path}")
