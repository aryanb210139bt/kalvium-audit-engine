"""
tests/test_audit_export.py
Tests for reports/audit_export.py — server-side XLSX/CSV generation for
the Audit History export feature. Covers the "EXPORT TESTS" checklist:
export all, export filtered, CSV/XLSX generated, correct row count,
correct headers, correct date/associate/combined filtering, and the
privacy requirement (no Email/Phone/Mobile in the output).
"""
import csv
import io
import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import openpyxl

import session_store
import activity_log
import history_filters as hf
import reports.audit_export as audit_export


def _fresh_dbs(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "DB_PATH", tmp_path / "sessions.db")
    monkeypatch.setattr(session_store, "_local", threading.local())
    session_store.init_db()
    monkeypatch.setattr(activity_log, "DB_PATH", tmp_path / "activity_log.db")
    monkeypatch.setattr(activity_log, "_local", threading.local())
    activity_log.init_db()
    import reports.associate_analytics as aa
    aa.invalidate_dashboard_cache()


def _mk(session_id, label, completed_at, lead_sheet=None, overall=70.0, grade="C",
        category_scores=None, top_strengths=None, improvement_areas=None):
    session_store.save_session(session_id, {
        "status": "completed", "label": label, "source_type": "csv_row",
        "source_url": f"https://drive.google.com/{session_id}",
        "created_at": completed_at, "completed_at": completed_at, "error": None,
        "lead_sheet": lead_sheet or {},
        "report": {
            "score": {"overall": overall, "grade": grade, "category_scores": category_scores or {}},
            "top_strengths": top_strengths or [], "improvement_areas": improvement_areas or [],
        },
    })


FULL_LEAD_SHEET = {
    "Email": "secret@example.com", "First Name": "Test", "Last Name": "Lead",
    "Lead Stage": "Prospect Lead", "Owner": "Abdul Kader Shanavas J",
    "Owner Email": "abdul.k@kalvium.com", "Phone Number": "9999999999",
    "Mobile Number": "8888888888", "Prospecting Owner": "Someone",
    "Prospect ID": "prospect-1", "Lead Number": "L-1", "Created On": "2026-07-01 09:00:00",
    "Lead Name": "Test Lead Full Name", "Lead link": "https://crm.example/lead/1",
    "Demo Date": "2026-08-05 12:00:00", "Lead Owner": "Abdul Kader Shanavas J",
    "TL Name": "Praveen GP", "Payment Done": "Yes",
}


def _seed_two(tmp_path, monkeypatch):
    _fresh_dbs(tmp_path, monkeypatch)
    _mk("s1", "Abdul Kader Shanavas J", "2026-08-10T10:00:00", lead_sheet=FULL_LEAD_SHEET,
        overall=82.5, grade="A",
        category_scores={
            "rapport_building": {"raw_score": 9.0, "category": "Opening & Rapport"},
            "closing_skills": {"raw_score": 3.0, "category": "KNET Pitch & CTA"},
        },
        top_strengths=["Great rapport", "Strong close"],
        improvement_areas=["Needs better discovery"])
    _mk("s2", "Kavya Talawar", "2026-08-20T10:00:00", overall=45.0, grade="D")


# ── CSV ───────────────────────────────────────────────────────────────────

def test_generate_csv_header_matches_spec_column_order(tmp_path, monkeypatch):
    _seed_two(tmp_path, monkeypatch)
    content, filename = audit_export.generate_csv()
    reader = csv.reader(io.StringIO(content.decode("utf-8-sig")))
    header = next(reader)
    assert header == audit_export.EXPORT_COLUMNS
    assert header[:5] == ["Audit Date", "Demo Date", "Associate", "Associate Email", "TL"]


def test_generate_csv_row_count_matches_total_not_a_page(tmp_path, monkeypatch):
    _fresh_dbs(tmp_path, monkeypatch)
    for i in range(45):  # more than one "page" (20) — export must not be page-limited
        _mk(f"s{i:02d}", f"Assoc{i}", f"2026-08-{(i%28)+1:02d}T00:00:00")
    content, _ = audit_export.generate_csv()
    rows = list(csv.reader(io.StringIO(content.decode("utf-8-sig"))))
    assert len(rows) - 1 == 45  # minus header


def test_generate_csv_no_pii_leaked(tmp_path, monkeypatch):
    _seed_two(tmp_path, monkeypatch)
    content, _ = audit_export.generate_csv()
    text = content.decode("utf-8-sig")
    assert "secret@example.com" not in text
    assert "9999999999" not in text
    assert "8888888888" not in text
    # But the CRM identifiers/associate-email the spec DOES want are present.
    assert "abdul.k@kalvium.com" in text
    assert "prospect-1" in text


def test_generate_csv_filename_format(tmp_path, monkeypatch):
    _fresh_dbs(tmp_path, monkeypatch)
    _, filename = audit_export.generate_csv()
    assert filename.startswith("kalvium_audit_export_")
    assert filename.endswith(".csv")


def test_generate_csv_filtered_excludes_non_matching(tmp_path, monkeypatch):
    _seed_two(tmp_path, monkeypatch)
    f = hf.parse_history_filters(associate="Abdul Kader Shanavas J")
    content, _ = audit_export.generate_csv(f)
    text = content.decode("utf-8-sig")
    assert "Abdul Kader Shanavas J" in text
    assert "Kavya Talawar" not in text


# ── XLSX ──────────────────────────────────────────────────────────────────

def test_generate_xlsx_has_two_sheets(tmp_path, monkeypatch):
    _seed_two(tmp_path, monkeypatch)
    content, filename = audit_export.generate_xlsx()
    wb = openpyxl.load_workbook(io.BytesIO(content))
    assert wb.sheetnames == ["Audit Results", "Export Summary"]
    assert filename.endswith(".xlsx")


def test_generate_xlsx_row_count_and_header(tmp_path, monkeypatch):
    _seed_two(tmp_path, monkeypatch)
    content, _ = audit_export.generate_xlsx()
    ws = openpyxl.load_workbook(io.BytesIO(content))["Audit Results"]
    assert ws.max_row == 3  # header + 2 data rows
    assert [c.value for c in ws[1]] == audit_export.EXPORT_COLUMNS


def test_generate_xlsx_freeze_panes_and_autofilter(tmp_path, monkeypatch):
    _seed_two(tmp_path, monkeypatch)
    content, _ = audit_export.generate_xlsx()
    ws = openpyxl.load_workbook(io.BytesIO(content))["Audit Results"]
    assert ws.freeze_panes == "A2"
    assert ws.auto_filter.ref == f"A1:AB3"


def _find_row_by_session_id(ws, header, session_id):
    """Rows sort newest-completed-first, so which physical row a given
    session lands on depends on the other seeded rows — locate by the
    Session ID column instead of assuming a fixed row index."""
    sid_col = header.index("Session ID")
    for r in range(2, ws.max_row + 1):
        if ws.cell(row=r, column=sid_col + 1).value == session_id:
            return r
    raise AssertionError(f"session {session_id} not found in export")


def test_generate_xlsx_category_scores_in_correct_columns(tmp_path, monkeypatch):
    _seed_two(tmp_path, monkeypatch)
    content, _ = audit_export.generate_xlsx()
    ws = openpyxl.load_workbook(io.BytesIO(content))["Audit Results"]
    header = [c.value for c in ws[1]]
    r = _find_row_by_session_id(ws, header, "s1")
    row = {header[i]: c.value for i, c in enumerate(ws[r])}
    assert row["Opening & Rapport"] == 9.0
    assert row["KNET Pitch & CTA"] == 3.0
    assert row["Strongest Category"] == "Opening & Rapport"
    assert row["Weakest Category"] == "KNET Pitch & CTA"
    assert row["Strengths"] == "Great rapport | Strong close"
    assert row["Areas to Improve"] == "Needs better discovery"


def test_generate_xlsx_hyperlinks_on_link_columns(tmp_path, monkeypatch):
    _seed_two(tmp_path, monkeypatch)
    content, _ = audit_export.generate_xlsx()
    ws = openpyxl.load_workbook(io.BytesIO(content))["Audit Results"]
    header = [c.value for c in ws[1]]
    r = _find_row_by_session_id(ws, header, "s1")
    lead_link_col = header.index("Lead Link") + 1
    cell = ws.cell(row=r, column=lead_link_col)
    assert cell.hyperlink is not None
    assert cell.hyperlink.target == "https://crm.example/lead/1"


def test_generate_xlsx_no_pii_leaked(tmp_path, monkeypatch):
    _seed_two(tmp_path, monkeypatch)
    content, _ = audit_export.generate_xlsx()
    ws = openpyxl.load_workbook(io.BytesIO(content))["Audit Results"]
    all_values = [str(c.value) for row in ws.iter_rows() for c in row if c.value is not None]
    assert not any("secret@example.com" in v for v in all_values)
    assert not any("9999999999" in v for v in all_values)


def test_generate_xlsx_summary_sheet_reflects_filters_and_dashboard(tmp_path, monkeypatch):
    _seed_two(tmp_path, monkeypatch)
    f = hf.parse_history_filters(associate="Abdul Kader Shanavas J")
    content, _ = audit_export.generate_xlsx(f)
    sw = openpyxl.load_workbook(io.BytesIO(content))["Export Summary"]
    lines = {row[0]: row[1] for row in sw.iter_rows(values_only=True) if row[0]}
    assert lines["Total Matching Audits"] == 1
    assert lines["Associate(s)"] == "Abdul Kader Shanavas J"
    assert lines["Average Score"] == 82.5
    assert lines["Most Common Grade"] == "A"


def test_generate_xlsx_empty_result_set_still_produces_valid_file(tmp_path, monkeypatch):
    _fresh_dbs(tmp_path, monkeypatch)
    f = hf.parse_history_filters(associate="Nobody Real")
    content, _ = audit_export.generate_xlsx(f)
    ws = openpyxl.load_workbook(io.BytesIO(content))["Audit Results"]
    assert ws.max_row == 1  # header only, no data rows, no crash


def test_export_filename_includes_single_associate_hint(tmp_path, monkeypatch):
    _fresh_dbs(tmp_path, monkeypatch)
    f = hf.parse_history_filters(associate="Abdul Kader Shanavas J")
    _, filename = audit_export.generate_xlsx(f)
    assert "Abdul_Kader_Shanavas_J" in filename


def test_export_filename_plain_when_unfiltered(tmp_path, monkeypatch):
    _fresh_dbs(tmp_path, monkeypatch)
    _, filename = audit_export.generate_xlsx(None)
    assert "kalvium_audit_export_" in filename
    assert filename.count("_") == 3  # kalvium_audit_export_YYYY-MM-DD.xlsx — no extra hint segment


# ── CATEGORY_EXPORT_ORDER derives from the real evaluation framework ────

def test_category_export_order_matches_evaluation_framework():
    from audit.evaluation_framework import EVALUATION_CATEGORIES
    expected = [(c["id"], c["label"]) for c in EVALUATION_CATEGORIES]
    assert audit_export.CATEGORY_EXPORT_ORDER == expected
    assert len(audit_export.CATEGORY_EXPORT_ORDER) == 9
