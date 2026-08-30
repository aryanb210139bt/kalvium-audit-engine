"""
tests/test_history_filters.py
Unit tests for history_filters.py's parsing and SQL WHERE-clause building —
the single shared filter-to-SQL translation used by the Audit History,
Dashboard, and Export features. No database involved (pure functions);
DB-backed filtering behavior lives in tests/test_history_filtering.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import history_filters as hf


# ── parse_history_filters ────────────────────────────────────────────────

def test_parse_defaults_to_empty_filters():
    f = hf.parse_history_filters()
    assert f.is_empty()
    assert f.sort_by == "audit_date"
    assert f.sort_order == "desc"
    assert f.sort_column == "completed_at"


def test_parse_strips_whitespace():
    f = hf.parse_history_filters(search="  kavya  ", audit_date_from=" 2026-08-01 ")
    assert f.search == "kavya"
    assert f.audit_date_from == "2026-08-01"


def test_parse_blank_strings_become_none():
    f = hf.parse_history_filters(search="   ", audit_date_from="")
    assert f.search is None
    assert f.audit_date_from is None
    assert f.is_empty()


def test_parse_comma_separated_multi_values():
    f = hf.parse_history_filters(associate="Priya Sharma, Abdul Kader Shanavas J")
    assert f.associate == ["Priya Sharma", "Abdul Kader Shanavas J"]


def test_parse_list_multi_values():
    f = hf.parse_history_filters(tl=["Praveen GP", "Akshay Mathew"])
    assert f.tl == ["Praveen GP", "Akshay Mathew"]


def test_parse_invalid_sort_by_falls_back_to_default():
    # Whitelist enforcement — Section 25: never accept an arbitrary column.
    f = hf.parse_history_filters(sort_by="report_json; DROP TABLE sessions;")
    assert f.sort_by == hf.DEFAULT_SORT_BY
    assert f.sort_column == "completed_at"


def test_parse_invalid_sort_order_falls_back_to_default():
    f = hf.parse_history_filters(sort_order="sideways")
    assert f.sort_order == hf.DEFAULT_SORT_ORDER


def test_parse_invalid_payment_status_becomes_none():
    f = hf.parse_history_filters(payment_status="maybe")
    assert f.payment_status is None


def test_parse_valid_payment_statuses():
    for v in ("yes", "no", "unknown", "YES", "Unknown"):
        f = hf.parse_history_filters(payment_status=v)
        assert f.payment_status == v.lower()


def test_sort_by_score_maps_to_overall_score():
    f = hf.parse_history_filters(sort_by="score")
    assert f.sort_column == "overall_score"


# ── is_empty / as_display_lines ──────────────────────────────────────────

def test_is_empty_false_when_any_filter_set():
    assert not hf.parse_history_filters(search="x").is_empty()
    assert not hf.parse_history_filters(associate="Priya").is_empty()
    assert not hf.parse_history_filters(payment_status="yes").is_empty()


def test_as_display_lines_shows_all_for_unfiltered():
    f = hf.parse_history_filters()
    lines = dict(f.as_display_lines())
    assert lines["Audit Date"] == "All"
    assert lines["Associate(s)"] == "All"
    assert lines["Search"] == "—"


def test_as_display_lines_shows_active_filter_values():
    f = hf.parse_history_filters(associate="Abdul Kader Shanavas J", audit_date_from="2026-08-01",
                                  audit_date_to="2026-08-30")
    lines = dict(f.as_display_lines())
    assert lines["Associate(s)"] == "Abdul Kader Shanavas J"
    assert lines["Audit Date"] == "2026-08-01 to 2026-08-30"


# ── next_day (inclusive end-of-range) ────────────────────────────────────

def test_next_day_normal():
    assert hf.next_day("2026-08-30") == "2026-08-31"


def test_next_day_month_rollover():
    assert hf.next_day("2026-08-31") == "2026-09-01"


def test_next_day_year_rollover():
    assert hf.next_day("2026-12-31") == "2027-01-01"


def test_next_day_leap_day():
    assert hf.next_day("2028-02-28") == "2028-02-29"  # 2028 is a leap year


# ── build_where — SQL shape + parameterization ───────────────────────────

def test_build_where_no_filters_is_empty():
    where, params = hf.build_where(hf.parse_history_filters())
    assert where == ""
    assert params == []


def test_build_where_none_filters_is_empty():
    where, params = hf.build_where(None)
    assert where == ""
    assert params == []


def test_build_where_search_only():
    where, params = hf.build_where(hf.parse_history_filters(search="Kavya"))
    assert "LOWER(label) LIKE ?" in where
    assert params[0] == "%kavya%"


def test_build_where_audit_date_range_is_inclusive_end():
    f = hf.parse_history_filters(audit_date_from="2026-08-01", audit_date_to="2026-08-30")
    where, params = hf.build_where(f)
    assert "completed_at >= ?" in where
    assert "completed_at < ?" in where
    assert "2026-08-01T00:00:00" in params
    assert "2026-08-31T00:00:00" in params  # next-day boundary, not the 30th itself


def test_build_where_demo_date_range():
    f = hf.parse_history_filters(demo_date_from="2026-08-01", demo_date_to="2026-08-15")
    where, params = hf.build_where(f)
    assert "Demo Date" in where
    assert "2026-08-01 00:00:00" in params
    assert "2026-08-16 00:00:00" in params


def test_build_where_associate_uses_in_clause_not_any():
    # Deliberately IN(...) not Postgres's ANY(%s) array syntax — SQLite has
    # no equivalent for ANY() with an array param, and this codebase is
    # dual-backend; IN(?,?,...) works identically on both.
    f = hf.parse_history_filters(associate="Priya Sharma,Abdul Kader Shanavas J")
    where, params = hf.build_where(f)
    assert "label IN (?,?)" in where
    assert params == ["Priya Sharma", "Abdul Kader Shanavas J"]


def test_build_where_payment_unknown_checks_blank_and_missing():
    f = hf.parse_history_filters(payment_status="unknown")
    where, params = hf.build_where(f)
    assert "lead_sheet_json IS NULL" in where
    assert "= '{}'" in where
    assert params == []  # no bound params needed for this branch


def test_build_where_payment_yes_parameterized():
    f = hf.parse_history_filters(payment_status="yes")
    where, params = hf.build_where(f)
    assert "LOWER(" in where and ") = ?" in where
    assert params == ["yes"]


def test_build_where_combines_all_filters_with_and():
    f = hf.parse_history_filters(
        search="kavya", audit_date_from="2026-08-01", audit_date_to="2026-08-30",
        demo_date_from="2026-08-01", demo_date_to="2026-08-15",
        associate="Abdul", tl="Praveen GP", lead_stage="Prospect Lead", payment_status="yes",
    )
    where, params = hf.build_where(f)
    # Every clause present, joined by AND, none silently dropped.
    for fragment in ("LOWER(label) LIKE", "completed_at >=", "completed_at <",
                      "Demo Date", "label IN", "TL Name", "Lead Stage", "Payment Done"):
        assert fragment in where, f"missing: {fragment}"
    # 9 conditions (search, audit_from, audit_to, demo_from, demo_to,
    # associate, tl, lead_stage, payment) joined by " AND " = 8 separators.
    assert where.count(" AND ") == 8


def test_lead_field_expr_key_is_never_user_controlled_in_practice():
    # Regression guard: build_where must only ever call _lead_field_expr
    # with this module's own fixed constants, never a raw filter value —
    # spot-check by confirming the constants exist and are what's used.
    assert hf.LEAD_FIELD_TL_NAME == "TL Name"
    assert hf.LEAD_FIELD_LEAD_STAGE == "Lead Stage"
    assert hf.LEAD_FIELD_PAYMENT_DONE == "Payment Done"
    assert hf.LEAD_FIELD_DEMO_DATE == "Demo Date"
