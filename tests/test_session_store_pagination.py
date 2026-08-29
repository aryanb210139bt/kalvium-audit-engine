"""
tests/test_session_store_pagination.py
Covers the server-side pagination + search added to session_store.py to
fix the Audit History page loading its entire dataset into the browser on
every visit (see list_sessions()/count()/get_reports_bulk()). Also covers
get_reports_bulk(), the bulk-fetch that replaced the N+1
get_report()-in-a-loop pattern used by both the history 'full=true'
response and the dashboard's category aggregation.
"""
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
