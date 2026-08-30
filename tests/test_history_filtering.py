"""
tests/test_history_filtering.py
DB-backed tests for the Audit History filter/export feature's storage
layer (session_store.list_sessions_filtered_page / list_sessions_matching
/ count_sessions_matching / distinct_associates / distinct_lead_sheet_values
/ iter_sessions_for_export), all built on history_filters.HistoryFilters —
covers the "API FILTER TESTS" checklist (no filters, each individual
filter, combined filters, invalid dates, empty filters, pagination,
sorting) at the storage layer, since this project's established pattern
(see tests/test_session_store_pagination.py) is to test these functions
directly rather than importing api.main in a pytest file — api.main's
module-level init_db() calls must never run before conftest.py's per-test
sqlite-forcing fixture, which only happens if api.main is imported inside
a test function, not at module collection time.
"""
import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import session_store
import history_filters as hf


def _fresh_db(tmp_path, monkeypatch, name="sessions.db"):
    monkeypatch.setattr(session_store, "DB_PATH", tmp_path / name)
    monkeypatch.setattr(session_store, "_local", threading.local())
    session_store.init_db()


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


def _seed(tmp_path, monkeypatch):
    """A small, realistic dataset spanning associates/TLs/stages/payment/
    dates, mirroring the real CSV-import shape (some sessions have no
    lead_sheet at all, matching real production: 37/60 sessions don't)."""
    _fresh_db(tmp_path, monkeypatch)
    _mk("s1", "Abdul Kader Shanavas J", "2026-08-05T10:00:00",
        lead_sheet={"TL Name": "Praveen GP", "Lead Stage": "Prospect Lead", "Payment Done": "Yes",
                    "Demo Date": "2026-08-01 12:00:00", "Prospect ID": "p1", "Lead Number": "L1",
                    "Lead Name": "Alpha Lead"})
    _mk("s2", "Kavya Talawar", "2026-08-15T10:00:00",
        lead_sheet={"TL Name": "Akshay Mathew", "Lead Stage": "Registered", "Payment Done": "No",
                    "Demo Date": "2026-08-10 12:00:00", "Prospect ID": "p2", "Lead Number": "L2",
                    "Lead Name": "Beta Lead"})
    _mk("s3", "Abdul Kader Shanavas J", "2026-08-25T10:00:00",
        lead_sheet={"TL Name": "Praveen GP", "Lead Stage": "Registered", "Payment Done": "Yes",
                    "Demo Date": "2026-08-20 12:00:00", "Prospect ID": "p3", "Lead Number": "L3",
                    "Lead Name": "Gamma Lead"})
    _mk("s4", "Kavya Talawar", "2026-09-01T10:00:00")  # no lead_sheet — matches real single-link uploads


# ── no filters / empty filters ───────────────────────────────────────────

def test_no_filters_returns_everything(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    rows, total = session_store.list_sessions_filtered_page(limit=20, filters=None, include_total=True)
    assert total == 4
    assert len(rows) == 4


def test_empty_filters_object_same_as_none(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    f = hf.parse_history_filters()
    rows, total = session_store.list_sessions_filtered_page(limit=20, filters=f, include_total=True)
    assert total == 4


# ── individual filters ───────────────────────────────────────────────────

def test_filter_by_associate(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    f = hf.parse_history_filters(associate="Abdul Kader Shanavas J")
    rows, total = session_store.list_sessions_filtered_page(limit=20, filters=f, include_total=True)
    assert total == 2
    assert {r["session_id"] for r in rows} == {"s1", "s3"}


def test_filter_by_tl(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    f = hf.parse_history_filters(tl="Praveen GP")
    rows, total = session_store.list_sessions_filtered_page(limit=20, filters=f, include_total=True)
    assert total == 2
    assert {r["session_id"] for r in rows} == {"s1", "s3"}


def test_filter_by_lead_stage(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    f = hf.parse_history_filters(lead_stage="Registered")
    rows, total = session_store.list_sessions_filtered_page(limit=20, filters=f, include_total=True)
    assert total == 2
    assert {r["session_id"] for r in rows} == {"s2", "s3"}


def test_filter_by_payment_status_yes(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    f = hf.parse_history_filters(payment_status="yes")
    rows, total = session_store.list_sessions_filtered_page(limit=20, filters=f, include_total=True)
    assert total == 2
    assert {r["session_id"] for r in rows} == {"s1", "s3"}


def test_filter_by_payment_status_unknown_includes_no_lead_sheet(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    f = hf.parse_history_filters(payment_status="unknown")
    rows, total = session_store.list_sessions_filtered_page(limit=20, filters=f, include_total=True)
    assert total == 1
    assert rows[0]["session_id"] == "s4"


def test_filter_by_audit_date_range_inclusive_end(tmp_path, monkeypatch):
    # s1=Aug 5, s2=Aug 15, s3=Aug 25 — range Aug 1-25 must include all
    # three, including s3 which lands exactly on the end date.
    _seed(tmp_path, monkeypatch)
    f = hf.parse_history_filters(audit_date_from="2026-08-01", audit_date_to="2026-08-25")
    rows, total = session_store.list_sessions_filtered_page(limit=20, filters=f, include_total=True)
    assert total == 3
    assert {r["session_id"] for r in rows} == {"s1", "s2", "s3"}


def test_filter_by_demo_date_range(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    f = hf.parse_history_filters(demo_date_from="2026-08-01", demo_date_to="2026-08-10")
    rows, total = session_store.list_sessions_filtered_page(limit=20, filters=f, include_total=True)
    assert total == 2
    assert {r["session_id"] for r in rows} == {"s1", "s2"}


def test_filter_by_search_matches_associate_lead_name_and_prospect_id(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    assert session_store.count_sessions_matching(hf.parse_history_filters(search="kavya")) == 2
    assert session_store.count_sessions_matching(hf.parse_history_filters(search="gamma")) == 1
    assert session_store.count_sessions_matching(hf.parse_history_filters(search="p2")) == 1


def test_invalid_date_format_does_not_crash(tmp_path, monkeypatch):
    # A garbage date string should not 500 — build_where treats it as a
    # plain string comparison (matches the rest of this app's convention
    # of storing timestamps as ISO text, not parsed dates), so an invalid
    # value just matches nothing rather than raising.
    _seed(tmp_path, monkeypatch)
    f = hf.parse_history_filters(audit_date_from="not-a-date")
    rows, total = session_store.list_sessions_filtered_page(limit=20, filters=f, include_total=True)
    assert isinstance(total, int)  # didn't raise


# ── combined filters ──────────────────────────────────────────────────────

def test_combined_associate_and_lead_stage(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    f = hf.parse_history_filters(associate="Abdul Kader Shanavas J", lead_stage="Registered")
    rows, total = session_store.list_sessions_filtered_page(limit=20, filters=f, include_total=True)
    assert total == 1
    assert rows[0]["session_id"] == "s3"


def test_combined_filters_that_match_nothing(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    f = hf.parse_history_filters(associate="Kavya Talawar", payment_status="yes")
    rows, total = session_store.list_sessions_filtered_page(limit=20, filters=f, include_total=True)
    assert total == 0
    assert rows == []


# ── pagination / sorting ──────────────────────────────────────────────────

def test_pagination_preserves_filters_across_pages(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch)
    for i in range(25):
        _mk(f"s{i:02d}", "Priya Sharma", f"2026-08-{i+1:02d}T00:00:00")
    _mk("other", "Someone Else", "2026-08-01T00:00:00")

    f = hf.parse_history_filters(associate="Priya Sharma")
    page1, total = session_store.list_sessions_filtered_page(limit=20, filters=f, include_total=True)
    assert total == 25
    last = page1[-1]
    cursor = (last[f.sort_column], last["session_id"])
    page2, total2 = session_store.list_sessions_filtered_page(limit=20, cursor=cursor, filters=f, include_total=False)
    assert total2 is None
    ids1 = {r["session_id"] for r in page1}
    ids2 = {r["session_id"] for r in page2}
    assert ids1.isdisjoint(ids2)
    assert len(ids1 | ids2) == 25
    assert "other" not in ids1 and "other" not in ids2


def test_default_sort_is_audit_date_descending(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    rows, _ = session_store.list_sessions_filtered_page(limit=20, filters=hf.parse_history_filters())
    assert [r["session_id"] for r in rows] == ["s4", "s3", "s2", "s1"]  # newest completed_at first


def test_sort_by_score_ascending(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch)
    _mk("low", "A", "2026-08-01T00:00:00", overall=10.0)
    _mk("high", "B", "2026-08-02T00:00:00", overall=90.0)
    f = hf.parse_history_filters(sort_by="score", sort_order="asc")
    rows, _ = session_store.list_sessions_filtered_page(limit=20, filters=f)
    assert [r["session_id"] for r in rows] == ["low", "high"]


# ── distinct filter-option values ─────────────────────────────────────────

def test_distinct_associates_excludes_blank_and_url_labels(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    _mk("url-only", "https://drive.google.com/abc", "2026-08-01T00:00:00")
    names = session_store.distinct_associates()
    assert set(names) == {"Abdul Kader Shanavas J", "Kavya Talawar"}


def test_distinct_lead_sheet_values(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    assert set(session_store.distinct_lead_sheet_values("TL Name")) == {"Praveen GP", "Akshay Mathew"}
    assert set(session_store.distinct_lead_sheet_values("Lead Stage")) == {"Prospect Lead", "Registered"}


# ── iter_sessions_for_export ─────────────────────────────────────────────

def test_iter_sessions_for_export_batches_and_includes_lead_sheet(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    batches = list(session_store.iter_sessions_for_export(filters=None, batch_size=2))
    all_ids = [r["session_id"] for b in batches for r in b]
    assert len(all_ids) == 4
    assert len(set(all_ids)) == 4  # no duplicates across batches
    assert len(batches) == 2  # 4 rows / batch_size 2

    by_id = {r["session_id"]: r for b in batches for r in b}
    assert by_id["s1"]["lead_sheet"]["TL Name"] == "Praveen GP"
    assert by_id["s4"]["lead_sheet"] == {}  # no lead_sheet — empty dict, not an error


def test_iter_sessions_for_export_respects_filters(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    f = hf.parse_history_filters(associate="Kavya Talawar")
    batches = list(session_store.iter_sessions_for_export(filters=f, batch_size=500))
    all_ids = [r["session_id"] for b in batches for r in b]
    assert set(all_ids) == {"s2", "s4"}


def test_iter_sessions_for_export_json_fields_are_parsed_not_raw_strings(tmp_path, monkeypatch):
    # Regression guard for the SQLite-vs-Postgres JSON shape quirk —
    # category_scores_json etc. must come back as real dict/list objects,
    # never a JSON-encoded string, regardless of backend.
    _fresh_db(tmp_path, monkeypatch)
    _mk("s1", "Alice", "2026-08-01T00:00:00",
        category_scores={"rapport_building": {"raw_score": 8.0, "category": "Opening & Rapport"}},
        top_strengths=["Great energy"], improvement_areas=["Needs closing"])
    batch = next(session_store.iter_sessions_for_export(filters=None, batch_size=10))
    rec = batch[0]
    assert isinstance(rec["category_scores_json"], dict)
    assert isinstance(rec["top_strengths_json"], list)
    assert rec["category_scores_json"]["rapport_building"]["raw_score"] == 8.0
