"""
reports/audit_excel_manager.py
Manages the master audit records Excel file.
Replicates the exact column structure and colour-coding of
'Audit Tracker AY - 2026.xlsx → 2026 Demo Auditng' sheet,
then appends AI-generated score columns after col 54.
"""
from __future__ import annotations
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

MASTER_PATH = Path("data/audit_records.xlsx")

# ── Exact column headers matching the original tracker ────────────────────────
# Source Of Webinar, Campaign level, the blank spacer column, and Whether
# pitched about webinar feedback were removed at the user's request (they
# were never actually used) — see the matching column deletions run once
# against the live Excel file and Google Sheet to keep everything aligned.
COLUMNS = [
    # ── CRM / Lead info  (yellow  FFF2CC) ─────────────────────────────────────
    "Audit date",                                          # 1
    "Demo Date",                                           # 2
    "Lead Link",                                           # 3
    "Phone Number",                                        # 4
    "Lead Owner",                                          # 5
    "TL Name",                                             # 6
    # ── Call info  (light-blue  C9DAF8) ───────────────────────────────────────
    "Call Duration",                                       # 7
    "Purpose Of Call\nExpained",                           # 8
    "Spoke with Whom ?",                                   # 9
    "Stream details collected and marked",                 # 10
    # ── Call quality  (pink  D5A6BD) ──────────────────────────────────────────
    "Pitched anything about intership from 2nd Year\n(Yes/No)",  # 11
    "Parent availability discussed? (Yes/No)",             # 12
    # ── Verdict / notes  (green + purple) ─────────────────────────────────────
    "Verdict",                                             # 13  green D9EAD3
    "\nDetailed review notes",                             # 14  purple 351C75
    "Auditor comments on Call Pitch",                      # 15  purple 351C75
    # ── Demo info  (green  D9EAD3) ────────────────────────────────────────────
    "Demo Link Available",                                 # 16
    "Demo Link",                                           # 17
    "Demo Duration",                                       # 18
    "Demo Attendees",                                      # 19
    "Camera Status",                                       # 20
    "counselling session attended through",                # 21  medium-green B6D7A8
    "Deck Presentation",                                   # 22  green D9EAD3
    "Screen-share Mode (Full screen / PiP)",               # 23
    "Counseller Intro",                                    # 24
    "Whether Rapport Built",                               # 25
    "All Slides Presented\n( Yes / No )",                  # 26
    # ── Content checks  (light-blue  A4C2F4) ──────────────────────────────────
    "Any Irrelevant University Pitched",                   # 27
    "Collage fees pitched",                                # 28
    "Whether Hostel fees and detailed Pitched",            # 29
    # ── Engagement  (orange  F9CB9C) ──────────────────────────────────────────
    "Explained about Working days and Timings of this Program",  # 30
    "Parent Engagement Level",                             # 31
    "Use of Stories/Examples",                             # 32
    # ── Quality checks  (pink  D5A6BD) ────────────────────────────────────────
    "Pitched anything about intership from 2nd Year\n(Yes/No)",  # 33  (intentional repeat – matches source)
    "Whether Explained about Internship Criteria",         # 34
    "Are the objectives of the demo clearly stated (Are they counselling properly without just reading from deck)",  # 35
    # ── Pitch quality  (light-blue  A4C2F4) ───────────────────────────────────
    "Whether Pushing/Stressing for Sale",                  # 36
    "Quality of Pitch",                                    # 37
    "Demo Pitch Verdict",                                  # 38
    # ── Post-demo  (green + pink + purple + green) ────────────────────────────
    "Payment Proof Availability\n( Yes / No )",            # 39  green D9EAD3
    "Father's Occupation",                                 # 40  pink  D5A6BD
    "Mother's Occupation",                                 # 41
    "Chance Of Admission",                                 # 42
    "Student's interest in CSE",                           # 43
    "Notes",                                               # 44  purple 351C75
    "Payment done",                                        # 45  green D9EAD3
    # ── Outcomes  (medium-blue  9FC5E8) ───────────────────────────────────────
    "Mode Of Sales",                                       # 46
    "Demo Grades(A/B/C)",                                  # 47
    "Final ARMC Conversion status",                        # 48
    "Parent Interaction Level",                            # 49
    "Mail Dropped",                                        # 50
    # ── AI-generated scores  (teal  B2DFDB) ──────────────────────────────────
    "Session ID",                                          # 51
    "Overall Score",                                       # 52
    "Grade",                                               # 53
    "Closing Skills Score",                                # 54
    "Product Explanation Score",                           # 55
    "Discovery Questions Score",                           # 56
    "Rapport Building Score",                              # 57
    "Fee Discussion Score",                                # 58
    "Two Way Communication Score",                         # 59
    "Placement Credibility Score",                         # 60
    "Trust Building Score",                                # 61
    "Follow Up Clarity Score",                             # 62
    "Deck Coverage Score",                                 # 63
    "MUST Covered",                                        # 64
    "SHOULD Covered",                                      # 65
]

# ── Header background colours per column (1-based index) ─────────────────────
def _header_bg(col_idx: int) -> tuple[str, str]:
    """Return (bg_hex, text_hex) for a header cell at 1-based col_idx."""
    if   col_idx <= 6:   return "FFF2CC", "000000"   # yellow
    elif col_idx <= 10:  return "C9DAF8", "000000"   # light blue
    elif col_idx <= 12:  return "D5A6BD", "000000"   # pink
    elif col_idx == 13:  return "D9EAD3", "000000"   # light green
    elif col_idx <= 15:  return "351C75", "FFFFFF"   # dark purple / white text
    elif col_idx <= 20:  return "D9EAD3", "000000"   # light green
    elif col_idx == 21:  return "B6D7A8", "000000"   # medium green
    elif col_idx <= 26:  return "D9EAD3", "000000"   # light green
    elif col_idx <= 29:  return "A4C2F4", "000000"   # light blue
    elif col_idx <= 32:  return "F9CB9C", "000000"   # orange
    elif col_idx <= 35:  return "D5A6BD", "000000"   # pink
    elif col_idx <= 38:  return "A4C2F4", "000000"   # light blue
    elif col_idx == 39:  return "D9EAD3", "000000"   # light green
    elif col_idx <= 43:  return "D5A6BD", "000000"   # pink
    elif col_idx == 44:  return "351C75", "FFFFFF"   # dark purple / white text
    elif col_idx == 45:  return "D9EAD3", "000000"   # light green
    elif col_idx <= 50:  return "9FC5E8", "000000"   # medium blue
    else:                return "B2DFDB", "004D40"   # teal (AI scores)


def _get_or_create_wb():
    """Load the master workbook or create it with styled headers if missing."""
    from openpyxl import Workbook, load_workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

    if MASTER_PATH.exists():
        return load_workbook(str(MASTER_PATH))

    MASTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    ws = wb.active
    ws.title = "2026 Demo Auditng"

    thin = Side(style="thin", color="D1D5DB")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    header_align = Alignment(horizontal="center", vertical="center", wrap_text=True)

    for col_idx, col_name in enumerate(COLUMNS, start=1):
        bg, fg = _header_bg(col_idx)
        cell = ws.cell(row=1, column=col_idx, value=col_name)
        cell.font      = Font(bold=True, color=fg, size=9)
        cell.fill      = PatternFill("solid", fgColor=bg)
        cell.alignment = header_align
        cell.border    = border
        ws.column_dimensions[cell.column_letter].width = max(12, min(30, len(col_name.replace("\n", "")) + 2))

    # Match original: I2 freeze, row height 45
    ws.row_dimensions[1].height = 45
    ws.freeze_panes = "A2"
    wb.save(str(MASTER_PATH))
    return wb


def append_audit_row(row_data: dict) -> int:
    """
    Write one audit row to the master Excel file. If this row's "Session ID"
    already exists in the sheet (i.e. this audit was pushed before and is
    being edited/re-pushed), that existing row is updated in place instead
    of appending a duplicate. row_data keys should match COLUMNS entries.
    Returns the row number written.
    """
    from openpyxl import load_workbook
    from openpyxl.styles import Alignment, PatternFill, Font, Border, Side

    wb = _get_or_create_wb()
    ws = wb.active

    session_id_col = COLUMNS.index("Session ID") + 1
    target_sid = str(row_data.get("Session ID", "")).strip()
    existing_row = None
    if target_sid:
        for r in range(2, ws.max_row + 1):
            if str(ws.cell(row=r, column=session_id_col).value or "").strip() == target_sid:
                existing_row = r
                break

    next_row = existing_row or (ws.max_row + 1)

    thin     = Side(style="thin", color="E5E7EB")
    bord     = Border(left=thin, right=thin, top=thin, bottom=thin)
    even_fill = PatternFill("solid", fgColor="F9FAFB")

    for col_idx, col_name in enumerate(COLUMNS, start=1):
        value = row_data.get(col_name, "")
        cell  = ws.cell(row=next_row, column=col_idx, value=value)
        cell.alignment = Alignment(vertical="center", wrap_text=False)
        cell.border    = bord
        if next_row % 2 == 0:
            cell.fill = even_fill

        # Colour-code grade cells
        if col_name in ("Demo Grades(A/B/C)", "Grade"):
            grade = str(value).strip().upper()
            color = "16A34A" if grade == "A" else "D97706" if grade == "B" else "DC2626"
            cell.font = Font(bold=True, color=color)

        # Colour Overall Score
        if col_name == "Overall Score":
            try:
                sc    = float(value)
                color = "16A34A" if sc >= 75 else "D97706" if sc >= 60 else "DC2626"
                cell.font = Font(bold=True, color=color)
            except (ValueError, TypeError):
                pass

    ws.row_dimensions[next_row].height = 18
    wb.save(str(MASTER_PATH))
    logger.info(f"{'Updated' if existing_row else 'Appended'} audit row {next_row} in {MASTER_PATH}")
    return next_row


def _demo_duration_category(seconds: float) -> str:
    """Convert seconds into the categorical values used by the original tracker."""
    mins = seconds / 60
    if mins >= 60:   return "More than an hour"
    if mins >= 45:   return "45 - 60 Mins"
    if mins >= 30:   return "30 - 45 Mins"
    return "Less than 30 mins"


def auto_fill_from_report(report: dict, session_meta: dict | None = None) -> dict:
    """
    Extract auto-fillable fields from an audit report dict.
    Keys match COLUMNS entries exactly.
    Manual/CRM-only fields with no upstream source are left as empty string.

    session_meta (optional): {"source_url", "label", "lead_sheet", "video_analysis"}
    from the upload session. "label" is the associate name; "lead_sheet" is the
    full, untouched CRM row from a batch CSV upload (api/main.py upload-csv) — any
    of its columns whose name matches a tracker COLUMN case-insensitively
    (e.g. its own "TL Name", "Demo Date", "Payment Done") overrides whatever
    this function would otherwise compute, since the CRM export is ground
    truth. Anything in lead_sheet that doesn't match a known column is not
    lost — it's just not placed into the tracker; see the audit detail page,
    which shows the full lead_sheet regardless of whether it mapped anywhere.
    "video_analysis" is the video_audit_store record for this session (or
    None if Video Snapshot Analysis was never run / hasn't finished yet) —
    when its status is "completed", its derived tracker_fields (Demo
    Attendees, Camera Status, Deck Presentation, Screen-share Mode) are used
    to fill those columns, which nothing else in this function ever sets.
    """
    session_meta = session_meta or {}
    video_rec = session_meta.get("video_analysis")
    video_tracker_fields = {}
    if video_rec and video_rec.get("status") == "completed":
        video_tracker_fields = (video_rec.get("summary") or {}).get("tracker_fields") or {}
    score      = report.get("score", {})
    cat_scores = score.get("category_scores", {})
    deck_cov   = report.get("deck_coverage") or {}
    coaching   = report.get("improvement_areas", [])
    strengths  = report.get("top_strengths", [])
    duration   = report.get("duration_seconds", 0)
    talk       = report.get("talk_ratio") or {}
    pi         = report.get("participant_intelligence") or {}
    attendance = pi.get("attendance") or {}

    overall = score.get("overall", 0)
    grade   = score.get("grade", "F")

    def cat(key):
        v = cat_scores.get(key)
        if isinstance(v, dict):
            return round(v.get("raw_score", 0), 1)
        return round(float(v), 1) if v is not None else ""

    # Demo grade A/B/C
    demo_grade = "A" if overall >= 75 else "B" if overall >= 60 else "C"

    # Duration — categorical value matching original tracker
    dur_cat = _demo_duration_category(duration)

    # Call duration string (HH:MM:SS style, matching original time cells)
    dur_h   = int(duration // 3600)
    dur_m   = int((duration % 3600) // 60)
    dur_s   = int(duration % 60)
    if dur_h:
        call_dur = f"{dur_h}:{dur_m:02d}:{dur_s:02d}"
    else:
        call_dur = f"{dur_m}:{dur_s:02d}"

    # Deck fields
    must_cov   = deck_cov.get("must_covered", "")
    must_tot   = deck_cov.get("must_total", "")
    shd_cov    = deck_cov.get("should_covered", "")
    shd_tot    = deck_cov.get("should_total", "")
    deck_score = deck_cov.get("coverage_score", "")
    deck_pct   = f"{deck_score:.0f}%" if deck_score != "" else ""
    must_str   = f"{must_cov}/{must_tot}" if must_tot else ""
    shd_str    = f"{shd_cov}/{shd_tot}" if shd_tot else ""

    def deck_covered(label_contains):
        for r in deck_cov.get("results", []):
            if label_contains.lower() in r.get("label", "").lower():
                return "Yes" if r.get("covered") else "No"
        return ""

    # Rapport
    rapport_score = cat("rapport_building")
    rapport = ("Yes" if isinstance(rapport_score, (int, float)) and rapport_score >= 6
               else "No" if isinstance(rapport_score, (int, float)) else "")

    # Quality of pitch — map 0-10 score to Good/Average/Poor
    def pitch_quality(sc):
        if not isinstance(sc, (int, float)):
            return ""
        if sc >= 7: return "Good"
        if sc >= 5: return "Average"
        return "Poor"

    closing = cat("closing_skills")
    product = cat("product_explanation")
    pitch_q = pitch_quality((closing + product) / 2 if isinstance(closing, (int,float)) and isinstance(product, (int,float)) else 0)

    notes_text     = " | ".join(coaching[:3]) if coaching else ""
    strengths_text = " | ".join(strengths[:2]) if strengths else ""

    # ── Session metadata (from the upload / lead sheet, not the transcript) ─
    lead_link   = session_meta.get("source_url", "") or ""
    lead_owner  = session_meta.get("label", "") or ""
    lead_sheet  = session_meta.get("lead_sheet") or {}
    lead_sheet_ci = {k.replace("\n", " ").strip().lower(): v for k, v in lead_sheet.items()}

    def _from_lead_sheet(column_name: str) -> str:
        """Case/whitespace-insensitive lookup of a tracker column against
        the raw CRM row, e.g. 'Payment done' matches the CSV's 'Payment Done'."""
        v = lead_sheet_ci.get(column_name.replace("\n", " ").strip().lower())
        return str(v).strip() if v else ""

    # ── TL Name — the CRM's own value wins; fall back to the uploaded
    # Associate→TL mapping only when the lead sheet doesn't supply one ──────
    tl_name = _from_lead_sheet("TL Name")
    if not tl_name and lead_owner:
        try:
            from reports.tl_mapping import lookup_tl
            tl_name = lookup_tl(lead_owner)
        except Exception:
            tl_name = ""

    # ── Participant Intelligence → "Spoke with Whom", parent fields ────────
    student_present = attendance.get("student_present")
    parent_present  = attendance.get("parent_present")
    spoke_with = ""
    if student_present is not None or parent_present is not None:
        who = ["Counsellor"]
        if student_present: who.append("Student")
        if parent_present:  who.append("Parent")
        spoke_with = ", ".join(who)

    parent_availability = ""
    if parent_present is not None:
        parent_availability = "Yes" if parent_present else "No"

    parent_engagement_level = ""
    if parent_present:
        pes = pi.get("parent_engagement_score")
        if isinstance(pes, (int, float)):
            parent_engagement_level = "High" if pes >= 7 else "Medium" if pes >= 4 else "Low"

    result = {
        "Audit date":                    datetime.now().strftime("%d/%m/%Y"),
        "Lead Link":                     lead_link,
        "Phone Number":                  _from_lead_sheet("Phone Number"),
        "Lead Owner":                    lead_owner,
        "TL Name":                       tl_name,
        "Spoke with Whom ?":             spoke_with,
        "Parent availability discussed? (Yes/No)": parent_availability,
        "Parent Engagement Level":       parent_engagement_level,
        "Demo Link Available":           "Yes" if lead_link else "",
        "Demo Link":                     lead_link,
        "Call Duration":                 call_dur,
        "Purpose Of Call\nExpained":     "Yes",
        "Whether Rapport Built":         rapport,
        "Verdict":                       f"{overall}/100 ({grade})",
        "\nDetailed review notes":        notes_text,
        "Auditor comments on Call Pitch": strengths_text,
        "Demo Duration":                 dur_cat,
        "counselling session attended through": "",   # manual
        "Deck Presentation":             "Yes" if deck_score and float(deck_score) > 0 else "No",
        "Counseller Intro":              deck_covered("introduction"),
        "All Slides Presented\n( Yes / No )": "Yes" if must_cov and must_tot and must_cov == must_tot else "No",
        "Collage fees pitched":          deck_covered("fee"),
        "Whether Hostel fees and detailed Pitched": deck_covered("hostel"),
        "Explained about Working days and Timings of this Program": deck_covered("work-integrated") or deck_covered("working"),
        "Pitched anything about intership from 2nd Year\n(Yes/No)": deck_covered("internship") or deck_covered("placement"),
        "Whether Explained about Internship Criteria": deck_covered("internship"),
        # Only set when Video Snapshot Analysis has run and completed —
        # nothing else derives these three; blank is correct until then.
        "Demo Attendees":                "",
        "Camera Status":                 "",
        "Screen-share Mode (Full screen / PiP)": "",
        "Are the objectives of the demo clearly stated (Are they counselling properly without just reading from deck)": "",
        "Whether Pushing/Stressing for Sale": "",
        "Quality of Pitch":              pitch_q,
        "Demo Pitch Verdict":            grade,
        "Demo Grades(A/B/C)":            demo_grade,
        # AI score columns
        "Session ID":                    report.get("session_id", ""),
        "Overall Score":                 overall,
        "Grade":                         grade,
        "Closing Skills Score":          cat("closing_skills"),
        "Product Explanation Score":     cat("product_explanation"),
        "Discovery Questions Score":     cat("discovery_questions"),
        "Rapport Building Score":        cat("rapport_building"),
        "Fee Discussion Score":          cat("fee_discussion"),
        "Two Way Communication Score":   cat("two_way_communication"),
        "Placement Credibility Score":   cat("placement_credibility"),
        "Trust Building Score":          cat("trust_building"),
        "Follow Up Clarity Score":       cat("follow_up_clarity"),
        "Deck Coverage Score":           deck_pct,
        "MUST Covered":                  must_str,
        "SHOULD Covered":                shd_str,
    }

    # Final pass: any tracker column with a same-named field in the raw CRM
    # row gets that value — the CRM export is ground truth and always wins
    # over a guess or a blank, for any column, not just the ones above.
    for col in COLUMNS:
        if not col:
            continue
        crm_val = _from_lead_sheet(col)
        if crm_val:
            result[col] = crm_val

    # Video Snapshot Analysis evidence — nothing else in this function ever
    # sets "Demo Attendees" / "Camera Status" / "Screen-share Mode", so these
    # only appear when the add-on ran. "Deck Presentation" already has a
    # deck-coverage-score-based guess above; real visual evidence (an actual
    # screenshot showing slides) is more reliable, so it wins here when
    # available — this is the one column the video add-on is allowed to
    # override rather than only fill in.
    if video_tracker_fields:
        for col in ("Demo Attendees", "Camera Status", "Screen-share Mode (Full screen / PiP)"):
            if video_tracker_fields.get(col):
                result[col] = video_tracker_fields[col]
        if "Deck Presentation" in video_tracker_fields:
            result["Deck Presentation"] = video_tracker_fields["Deck Presentation"]

    return result
