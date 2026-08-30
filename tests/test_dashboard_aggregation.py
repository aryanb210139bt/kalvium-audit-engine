"""
tests/test_dashboard_aggregation.py
Covers reports/associate_analytics.py's backend aggregation for the Audit
Dashboard (Category Performance / Strengths & Improvements / Trends /
Recent vs Previous) — this is the lightweight backend rollup that replaced
fetching every historical session into the browser to compute those stats
client-side. Also covers the short-lived dashboard cache and its
invalidation.
"""
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import activity_log
import session_store
import history_filters as hf
import reports.associate_analytics as associate_analytics


def _fresh_dbs(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "DB_PATH", tmp_path / "sessions.db")
    monkeypatch.setattr(session_store, "_local", threading.local())
    session_store.init_db()

    monkeypatch.setattr(activity_log, "DB_PATH", tmp_path / "activity_log.db")
    monkeypatch.setattr(activity_log, "_local", threading.local())
    activity_log.init_db()

    # Every test starts with a clean, un-expired-into cache so one test's
    # aggregation result can't leak into the next.
    associate_analytics.invalidate_dashboard_cache()


def _mk(session_id, label, created_at, overall, grade, category_scores=None, lead_sheet=None):
    session_store.save_session(session_id, {
        "status": "completed",
        "label": label,
        "source_type": "upload",
        "source_url": "",
        "created_at": created_at,
        "completed_at": created_at,
        "error": None,
        "lead_sheet": lead_sheet or {},
        "report": {
            "score": {"overall": overall, "grade": grade, "category_scores": category_scores or {}},
            "top_strengths": [],
            "improvement_areas": [],
        },
    })


def test_build_overall_dashboard_basic_stats(tmp_path, monkeypatch):
    _fresh_dbs(tmp_path, monkeypatch)
    _mk("s1", "Alice", "2026-01-01T00:00:00", 90, "A")
    _mk("s2", "Bob",   "2026-01-02T00:00:00", 70, "C")

    d = associate_analytics.build_overall_dashboard()
    assert d["total_audits"] == 2
    assert d["average_score"] == 80.0
    assert d["grade_distribution"] == {"A": 1, "C": 1}
    assert len(d["trend"]) == 2


def test_category_breakdown_averages_across_sessions(tmp_path, monkeypatch):
    _fresh_dbs(tmp_path, monkeypatch)
    _mk("s1", "Alice", "2026-01-01T00:00:00", 90, "A",
        category_scores={"compliance": {"raw_score": 80, "category": "Compliance"}})
    _mk("s2", "Bob", "2026-01-02T00:00:00", 70, "C",
        category_scores={"compliance": {"raw_score": 60, "category": "Compliance"}})

    d = associate_analytics.build_overall_dashboard()
    assert d["category_breakdown"]["Compliance"]["average"] == 70.0  # (80+60)/2
    assert d["category_breakdown"]["Compliance"]["sessions"] == 2


def test_dashboard_uses_reports_bulk_not_per_session_calls(tmp_path, monkeypatch):
    # Regression guard for the N+1 fix: build_overall_dashboard() must
    # fetch report bodies via ONE bulk call, not one get_report() call per
    # session. This is the actual root cause of the slow dashboard load.
    _fresh_dbs(tmp_path, monkeypatch)
    for i in range(10):
        _mk(f"s{i}", f"Assoc{i}", f"2026-01-{i+1:02d}T00:00:00", 80, "B")

    calls = {"get_report": 0, "get_reports_bulk": 0}
    orig_get_report = session_store.get_report
    orig_bulk = session_store.get_reports_bulk

    def counted_get_report(sid):
        calls["get_report"] += 1
        return orig_get_report(sid)

    def counted_bulk(ids):
        calls["get_reports_bulk"] += 1
        return orig_bulk(ids)

    monkeypatch.setattr(session_store, "get_report", counted_get_report)
    monkeypatch.setattr(session_store, "get_reports_bulk", counted_bulk)

    associate_analytics.build_overall_dashboard()

    assert calls["get_report"] == 0        # no per-session round trips
    assert calls["get_reports_bulk"] == 1  # exactly one bulk fetch


def test_dashboard_result_is_cached_within_ttl(tmp_path, monkeypatch):
    _fresh_dbs(tmp_path, monkeypatch)
    _mk("s1", "Alice", "2026-01-01T00:00:00", 90, "A")

    first = associate_analytics.build_overall_dashboard()

    # Add a new session directly, bypassing invalidate_dashboard_cache() —
    # a cached result should NOT reflect it yet.
    _mk("s2", "Bob", "2026-01-02T00:00:00", 50, "F")
    second = associate_analytics.build_overall_dashboard()

    assert first == second
    assert second["total_audits"] == 1  # still the stale cached value


def test_invalidate_dashboard_cache_forces_recompute(tmp_path, monkeypatch):
    _fresh_dbs(tmp_path, monkeypatch)
    _mk("s1", "Alice", "2026-01-01T00:00:00", 90, "A")
    associate_analytics.build_overall_dashboard()

    _mk("s2", "Bob", "2026-01-02T00:00:00", 50, "F")
    associate_analytics.invalidate_dashboard_cache()

    fresh = associate_analytics.build_overall_dashboard()
    assert fresh["total_audits"] == 2  # reflects the new session now


def test_dashboard_cache_expires_after_ttl(tmp_path, monkeypatch):
    _fresh_dbs(tmp_path, monkeypatch)
    _mk("s1", "Alice", "2026-01-01T00:00:00", 90, "A")
    associate_analytics.build_overall_dashboard()

    _mk("s2", "Bob", "2026-01-02T00:00:00", 50, "F")
    # Simulate TTL having elapsed without calling invalidate() explicitly.
    associate_analytics._dashboard_cache["expires_at"] = 0.0

    fresh = associate_analytics.build_overall_dashboard()
    assert fresh["total_audits"] == 2


def test_empty_history_does_not_error(tmp_path, monkeypatch):
    _fresh_dbs(tmp_path, monkeypatch)
    d = associate_analytics.build_overall_dashboard()
    assert d["total_audits"] == 0
    assert d["average_score"] is None
    assert d["trend"] == []


def test_trend_label_prefers_tracker_lead_owner_over_upload_label(tmp_path, monkeypatch):
    # _load_sessions() overrides the upload-time label with the tracker
    # push's Lead Owner when one exists — this must survive the bulk-fetch
    # refactor unchanged (existing behavior, not something this task should
    # alter).
    _fresh_dbs(tmp_path, monkeypatch)
    _mk("s1", "raw-upload-label", "2026-01-01T00:00:00", 90, "A")
    activity_log.log_event("pushed_to_tracker", session_id="s1", associate="Priya Sharma",
                            actor="pytest", detail={"fields": {"Lead Owner": "Priya Sharma"}})

    d = associate_analytics.build_overall_dashboard()
    assert d["trend"][0]["label"] == "Priya Sharma"


# ── Filtered dashboard (Section 16: "History filtering ≠ Dashboard
# filtering ≠ Export filtering" must never happen — one shared
# history_filters.build_where used by all three) ────────────────────────

def test_dashboard_with_associate_filter_only_counts_that_associate(tmp_path, monkeypatch):
    _fresh_dbs(tmp_path, monkeypatch)
    _mk("s1", "Abdul", "2026-01-01T00:00:00", 90, "A")
    _mk("s2", "Abdul", "2026-01-02T00:00:00", 70, "C")
    _mk("s3", "Kavya", "2026-01-03T00:00:00", 50, "F")

    d = associate_analytics.build_overall_dashboard(hf.parse_history_filters(associate="Abdul"))
    assert d["total_audits"] == 2
    assert d["average_score"] == 80.0  # (90+70)/2 — Kavya's 50 excluded


def test_dashboard_with_date_filter_narrows_trend(tmp_path, monkeypatch):
    _fresh_dbs(tmp_path, monkeypatch)
    _mk("s1", "Abdul", "2026-08-01T00:00:00", 90, "A")
    _mk("s2", "Abdul", "2026-08-15T00:00:00", 70, "C")
    _mk("s3", "Abdul", "2026-09-01T00:00:00", 50, "F")

    f = hf.parse_history_filters(audit_date_from="2026-08-01", audit_date_to="2026-08-31")
    d = associate_analytics.build_overall_dashboard(f)
    assert d["total_audits"] == 2
    assert d["average_score"] == 80.0


def test_dashboard_with_combined_filters(tmp_path, monkeypatch):
    _fresh_dbs(tmp_path, monkeypatch)
    _mk("s1", "Abdul", "2026-08-01T00:00:00", 90, "A", lead_sheet={"TL Name": "Praveen GP"})
    _mk("s2", "Abdul", "2026-08-15T00:00:00", 70, "C", lead_sheet={"TL Name": "Akshay Mathew"})
    _mk("s3", "Kavya", "2026-08-20T00:00:00", 50, "F", lead_sheet={"TL Name": "Praveen GP"})

    f = hf.parse_history_filters(associate="Abdul", tl="Praveen GP")
    d = associate_analytics.build_overall_dashboard(f)
    assert d["total_audits"] == 1
    assert d["average_score"] == 90.0


def test_dashboard_filters_match_history_filters_exactly(tmp_path, monkeypatch):
    # The actual point of the shared history_filters module: the dashboard
    # and the history list must agree on which sessions match a given
    # filter — count them independently and compare.
    _fresh_dbs(tmp_path, monkeypatch)
    _mk("s1", "Abdul", "2026-08-01T00:00:00", 90, "A", lead_sheet={"Lead Stage": "Registered"})
    _mk("s2", "Abdul", "2026-08-02T00:00:00", 70, "C", lead_sheet={"Lead Stage": "Prospect Lead"})
    _mk("s3", "Kavya", "2026-08-03T00:00:00", 50, "F", lead_sheet={"Lead Stage": "Registered"})

    f = hf.parse_history_filters(lead_stage="Registered")
    dashboard_total = associate_analytics.build_overall_dashboard(f)["total_audits"]
    history_total = session_store.count_sessions_matching(f)
    assert dashboard_total == history_total == 2


def test_dashboard_cache_not_shared_across_different_filters(tmp_path, monkeypatch):
    # A filtered dashboard must never be served from (or pollute) the
    # unfiltered dashboard's 30s cache slot.
    _fresh_dbs(tmp_path, monkeypatch)
    _mk("s1", "Abdul", "2026-01-01T00:00:00", 90, "A")
    _mk("s2", "Kavya", "2026-01-02T00:00:00", 50, "F")

    unfiltered = associate_analytics.build_overall_dashboard()
    assert unfiltered["total_audits"] == 2

    filtered = associate_analytics.build_overall_dashboard(hf.parse_history_filters(associate="Abdul"))
    assert filtered["total_audits"] == 1

    # Re-fetching unfiltered afterward still returns the cached unfiltered
    # result, not something contaminated by the filtered call in between.
    unfiltered_again = associate_analytics.build_overall_dashboard()
    assert unfiltered_again["total_audits"] == 2
