"""
tests/test_session_store_pagination.py
Covers the server-side pagination + search added to session_store.py to
fix the Audit History page loading its entire dataset into the browser on
every visit (see list_sessions()/count()/get_reports_bulk()). Also covers
get_reports_bulk(), the bulk-fetch that replaced the N+1
get_report()-in-a-loop pattern used by both the history 'full=true'
response and the dashboard's category aggregation.
"""
import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import session_store


def _fresh_db(tmp_path, monkeypatch, name="sessions.db"):
    monkeypatch.setattr(session_store, "DB_PATH", tmp_path / name)
    monkeypatch.setattr(session_store, "_local", threading.local())
    session_store.init_db()


def _mk(session_id, label, created_at, overall=80.0, grade="B", report=None):
    session_store.save_session(session_id, {
        "status": "completed",
        "label": label,
        "source_type": "upload",
        "source_url": "",
        "created_at": created_at,
        "completed_at": created_at,
        "error": None,
        "lead_sheet": {},
        "report": report or {"score": {"overall": overall, "grade": grade, "category_scores": {}}},
    })


def test_list_sessions_orders_newest_first(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch)
    _mk("s1", "Alice", "2026-01-01T00:00:00")
    _mk("s2", "Bob",   "2026-01-03T00:00:00")
    _mk("s3", "Carol", "2026-01-02T00:00:00")

    rows = session_store.list_sessions(limit=10)
    assert [r["session_id"] for r in rows] == ["s2", "s3", "s1"]


def test_list_sessions_pagination_limit_offset(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch)
    for i in range(25):
        _mk(f"s{i:02d}", f"Assoc{i}", f"2026-01-{i+1:02d}T00:00:00")

    page1 = session_store.list_sessions(limit=20, offset=0)
    page2 = session_store.list_sessions(limit=20, offset=20)

    assert len(page1) == 20
    assert len(page2) == 5
    # No overlap between pages, and together they cover every session exactly once.
    ids1 = {r["session_id"] for r in page1}
    ids2 = {r["session_id"] for r in page2}
    assert ids1.isdisjoint(ids2)
    assert len(ids1 | ids2) == 25


def test_count_matches_total_rows(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch)
    for i in range(7):
        _mk(f"s{i}", f"Assoc{i}", f"2026-01-0{i+1}T00:00:00")
    assert session_store.count() == 7


def test_search_filters_by_label_case_insensitive(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch)
    _mk("s1", "Priya Sharma", "2026-01-01T00:00:00")
    _mk("s2", "Rahul Verma",  "2026-01-02T00:00:00")

    rows = session_store.list_sessions(limit=10, search="priya")  # lowercase, should still match "Priya"
    assert [r["session_id"] for r in rows] == ["s1"]
    assert session_store.count(search="priya") == 1


def test_search_filters_by_session_id(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch)
    _mk("session-abc123", "Priya Sharma", "2026-01-01T00:00:00")
    _mk("session-xyz789", "Rahul Verma",  "2026-01-02T00:00:00")

    rows = session_store.list_sessions(limit=10, search="abc123")
    assert [r["session_id"] for r in rows] == ["session-abc123"]


def test_search_no_match_returns_empty(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch)
    _mk("s1", "Priya Sharma", "2026-01-01T00:00:00")
    assert session_store.list_sessions(limit=10, search="nonexistent") == []
    assert session_store.count(search="nonexistent") == 0


def test_search_respects_pagination(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch)
    for i in range(15):
        _mk(f"s{i:02d}", "Priya Sharma", f"2026-01-{i+1:02d}T00:00:00")
    _mk("other", "Someone Else", "2026-02-01T00:00:00")

    total = session_store.count(search="priya")
    assert total == 15
    page1 = session_store.list_sessions(limit=10, offset=0, search="priya")
    page2 = session_store.list_sessions(limit=10, offset=10, search="priya")
    assert len(page1) == 10
    assert len(page2) == 5


def test_list_sessions_omits_report_json_blob(tmp_path, monkeypatch):
    # The whole point of list_sessions() (vs. get_session()) is that it's
    # lightweight — no report_json blob — so listing pages stays cheap
    # regardless of how large individual reports are.
    _fresh_db(tmp_path, monkeypatch)
    _mk("s1", "Alice", "2026-01-01T00:00:00")
    row = session_store.list_sessions(limit=10)[0]
    assert "report_json" not in row
    assert "report" not in row


def test_get_reports_bulk_returns_only_requested_sessions_with_reports(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch)
    _mk("s1", "Alice", "2026-01-01T00:00:00",
        report={"score": {"overall": 90, "grade": "A", "category_scores": {"compliance": {"raw_score": 85, "category": "Compliance"}}}})
    _mk("s2", "Bob", "2026-01-02T00:00:00",
        report={"score": {"overall": 70, "grade": "C", "category_scores": {}}})
    _mk("s3", "Carol", "2026-01-03T00:00:00",
        report={"score": {"overall": 60, "grade": "D", "category_scores": {}}})

    result = session_store.get_reports_bulk(["s1", "s3", "does-not-exist"])
    assert set(result.keys()) == {"s1", "s3"}
    assert result["s1"]["score"]["overall"] == 90
    assert result["s1"]["score"]["category_scores"]["compliance"]["raw_score"] == 85
    assert "s2" not in result  # not requested


def test_get_reports_bulk_empty_input_returns_empty_dict(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch)
    assert session_store.get_reports_bulk([]) == {}


def test_get_reports_bulk_matches_get_report_per_session(tmp_path, monkeypatch):
    # Correctness guarantee: bulk fetch must return identical data to what
    # the old per-session get_report() loop would have produced.
    _fresh_db(tmp_path, monkeypatch)
    for i in range(5):
        _mk(f"s{i}", f"Assoc{i}", f"2026-01-0{i+1}T00:00:00",
            report={"score": {"overall": 50 + i, "grade": "B", "category_scores": {}}})

    ids = [f"s{i}" for i in range(5)]
    bulk = session_store.get_reports_bulk(ids)
    for sid in ids:
        assert bulk[sid] == session_store.get_report(sid)


# ── list_sessions_page — the combined query that replaced 3 sequential
# round trips (list_sessions + get_reports_bulk + count) with 1, for the
# history page specifically. See session_store.py's docstring for the
# full before/after story.

def _mk_with_summary(session_id, label, created_at, overall=80.0, grade="B",
                      category_scores=None, top_strengths=None, improvement_areas=None):
    _mk(session_id, label, created_at, overall, grade, report={
        "score": {"overall": overall, "grade": grade, "category_scores": category_scores or {}},
        "top_strengths": top_strengths or [],
        "improvement_areas": improvement_areas or [],
    })


def test_list_sessions_page_first_page_includes_total(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch)
    for i in range(25):
        _mk(f"s{i:02d}", f"Assoc{i}", f"2026-01-{i+1:02d}T00:00:00")

    rows, total = session_store.list_sessions_page(limit=20, include_total=True)
    assert len(rows) == 20
    assert total == 25
    assert [r["session_id"] for r in rows] == [f"s{i:02d}" for i in range(24, 4, -1)]  # newest first


def test_list_sessions_page_cursor_continues_without_gap_or_overlap(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch)
    for i in range(25):
        _mk(f"s{i:02d}", f"Assoc{i}", f"2026-01-{i+1:02d}T00:00:00")

    page1, total = session_store.list_sessions_page(limit=20, include_total=True)
    last = page1[-1]
    cursor = (last["created_at"], last["session_id"])
    page2, total2 = session_store.list_sessions_page(limit=20, cursor=cursor, include_total=False)

    assert total2 is None  # not recomputed on a "load more" page
    ids1 = {r["session_id"] for r in page1}
    ids2 = {r["session_id"] for r in page2}
    assert ids1.isdisjoint(ids2)          # no duplicate rows across pages
    assert len(ids1 | ids2) == 25         # every row covered exactly once
    assert len(page2) == 5


def test_list_sessions_page_cursor_ties_broken_by_session_id(tmp_path, monkeypatch):
    # Multiple sessions can legitimately share the same created_at (e.g. a
    # batch upload) — ORDER BY created_at DESC, session_id DESC plus the
    # (created_at, session_id) < (?, ?) cursor condition must still walk
    # through every one of them exactly once, not skip or repeat any.
    _fresh_db(tmp_path, monkeypatch)
    same_time = "2026-01-01T00:00:00"
    for i in range(5):
        _mk(f"tie{i}", f"Assoc{i}", same_time)

    page1, total = session_store.list_sessions_page(limit=2, include_total=True)
    assert total == 5
    last = page1[-1]
    cursor = (last["created_at"], last["session_id"])
    page2, _ = session_store.list_sessions_page(limit=2, cursor=cursor)
    last2 = page2[-1]
    cursor2 = (last2["created_at"], last2["session_id"])
    page3, _ = session_store.list_sessions_page(limit=2, cursor=cursor2)

    all_ids = [r["session_id"] for r in page1 + page2 + page3]
    assert sorted(all_ids) == sorted(f"tie{i}" for i in range(5))
    assert len(set(all_ids)) == 5  # no duplicates despite identical created_at


def _normalize_json_field(value):
    # Mirrors api/main.py's _parse_json_field(): SQLite's json_extract()
    # returns a JSON-encoded string for objects/arrays, Postgres's jsonb
    # cast comes back already parsed. list_sessions_page() deliberately
    # returns the backend-native shape (that normalization is the API
    # layer's job, not the store's) — so these tests normalize the same
    # way api/main.py does before asserting on the contents.
    if value is None or isinstance(value, (dict, list)):
        return value
    return json.loads(value)


def test_list_sessions_page_summary_fields_extracted_correctly(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch)
    _mk_with_summary("s1", "Alice", "2026-01-01T00:00:00",
                      category_scores={"compliance": {"raw_score": 8.5, "category": "Compliance"}},
                      top_strengths=["Great rapport"], improvement_areas=["Needs closing practice"])

    rows, _ = session_store.list_sessions_page(limit=10, include_summary=True)
    r = rows[0]
    assert _normalize_json_field(r["category_scores_json"])["compliance"]["raw_score"] == 8.5
    assert _normalize_json_field(r["top_strengths_json"]) == ["Great rapport"]
    assert _normalize_json_field(r["improvement_areas_json"]) == ["Needs closing practice"]


def test_list_sessions_page_without_summary_omits_report_json_cost(tmp_path, monkeypatch):
    # include_summary=False must not even attempt the JSON columns — used
    # by callers that only need the lightweight row fields.
    _fresh_db(tmp_path, monkeypatch)
    _mk("s1", "Alice", "2026-01-01T00:00:00")
    rows, _ = session_store.list_sessions_page(limit=10, include_summary=False)
    assert "category_scores_json" not in rows[0]


def test_list_sessions_page_search_combines_with_cursor(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch)
    for i in range(15):
        _mk(f"s{i:02d}", "Priya Sharma", f"2026-01-{i+1:02d}T00:00:00")
    _mk("other", "Someone Else", "2026-02-01T00:00:00")

    page1, total = session_store.list_sessions_page(limit=10, search="priya", include_total=True)
    assert total == 15
    assert all("Priya" in r["label"] for r in page1)

    last = page1[-1]
    cursor = (last["created_at"], last["session_id"])
    page2, _ = session_store.list_sessions_page(limit=10, cursor=cursor, search="priya")
    assert len(page2) == 5
    assert all("Priya" in r["label"] for r in page2)
    assert "other" not in {r["session_id"] for r in page2}


def test_list_sessions_page_empty_table(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch)
    rows, total = session_store.list_sessions_page(limit=20, include_total=True)
    assert rows == []
    assert total == 0


def test_list_sessions_page_no_report_leaves_summary_fields_none(tmp_path, monkeypatch):
    # A session with no report_json at all (e.g. failed before scoring)
    # must not error the JSON extraction — fields come back None/empty.
    _fresh_db(tmp_path, monkeypatch)
    session_store.save_session("s1", {
        "status": "failed", "label": "Alice", "source_type": "upload", "source_url": "",
        "created_at": "2026-01-01T00:00:00", "completed_at": "", "error": "boom",
        "lead_sheet": {}, "report": None,
    })
    rows, _ = session_store.list_sessions_page(limit=10, include_summary=True)
    assert rows[0]["category_scores_json"] is None
    assert rows[0]["top_strengths_json"] is None
    assert rows[0]["improvement_areas_json"] is None
