"""
reports/associate_analytics.py
Aggregates completed sessions by associate into per-associate profiles:
session count, average score, grade distribution, and score trend over time.

Which associate a session belongs to is decided by, in priority order:
  1. The "Lead Owner" from that session's most recent tracker push — this is
     the deliberate, roster-validated name picked via the Associate picker
     when pushing to the tracker (see activity_log.latest_tracker_lead_owner_by_session).
  2. The upload-time label, as a fallback for sessions never pushed yet.
Session 1 is authoritative once it exists — a session pushed under "Priya
Sharma" belongs to Priya Sharma here even if it was uploaded unlabeled.

Purely additive. Reads from session_store (SQLite — see session_store.py)
for the session list and, for category-level detail, each session's full
report. Does not touch the audit/evaluation pipeline in any way.
"""
from __future__ import annotations
import time
import activity_log
import session_store

UNASSIGNED = "Unassigned"

# ── Short-lived cache for the overall dashboard ─────────────────────────────
# build_overall_dashboard() aggregates across every session — not cheap, and
# not user-specific (it's the same admin-only rollup for everyone, no
# per-viewer variation), so a brief in-process cache is safe: it just means
# a new audit's effect on the dashboard can take up to _DASHBOARD_CACHE_TTL
# seconds to show up, in exchange for not recomputing it from scratch on
# every page load/poll. Single-process deployment (confirmed: no --workers/
# gunicorn anywhere) — a plain module-level dict is enough, no need for a
# shared/distributed cache.
_DASHBOARD_CACHE_TTL = 30  # seconds
_dashboard_cache: dict = {"data": None, "expires_at": 0.0}


def invalidate_dashboard_cache() -> None:
    """Call after anything that changes the audit history (a new session
    saved, deleted, etc.) if you want the next dashboard load to reflect
    it immediately rather than waiting out the TTL. Not required for
    correctness — the cache expires on its own — just avoids the up-to-
    30s staleness window when it matters."""
    _dashboard_cache["data"] = None
    _dashboard_cache["expires_at"] = 0.0


def _is_real_label(label: str) -> bool:
    """A 'label' only counts as an associate name if it isn't blank and
    isn't the auto-generated fallback (a raw source URL)."""
    label = (label or "").strip()
    if not label:
        return False
    if label.lower().startswith(("http://", "https://")):
        return False
    return True


def _load_sessions(filters=None) -> list[dict]:
    """Session summaries from session_store, remapped to the field names
    the rest of this module already expects (date/session_id/label/...) —
    with the label overridden by the tracker-push Lead Owner where one exists.

    `filters` (a history_filters.HistoryFilters, or None for "everything")
    is the SAME shared filter object the Audit History and Export features
    use (history_filters.py / session_store.list_sessions_matching) — so
    "dashboard responds to filters" means literally the same WHERE clause,
    never a second reimplementation that could drift out of sync."""
    rows = session_store.list_sessions_matching(filters)
    lead_owners = activity_log.latest_tracker_lead_owner_by_session()
    return [{
        "session_id":       r["session_id"],
        "date":             r["created_at"],
        "label":            lead_owners.get(r["session_id"]) or r["label"],
        "overall_score":    r["overall_score"],
        "grade":            r["grade"],
    } for r in rows]


def _grade_bucket(grade: str) -> str:
    return (grade or "").replace("+", "").upper() or "?"


def build_associate_profiles(filters=None) -> list[dict]:
    """One row per associate, named associates first (most sessions first),
    with anything unlabeled or URL-only grouped into a single 'Unassigned'
    row at the end."""
    sessions = _load_sessions(filters)
    groups: dict[str, list[dict]] = {}
    for s in sessions:
        label = s.get("label", "")
        key = label.strip() if _is_real_label(label) else UNASSIGNED
        groups.setdefault(key, []).append(s)

    profiles = []
    for name, sess_list in groups.items():
        sess_sorted = sorted(sess_list, key=lambda s: s.get("date", ""))
        scores = [s.get("overall_score") for s in sess_sorted if isinstance(s.get("overall_score"), (int, float))]
        grades: dict[str, int] = {}
        for s in sess_sorted:
            g = _grade_bucket(s.get("grade", ""))
            grades[g] = grades.get(g, 0) + 1
        trend = [{"date": (s.get("date") or "")[:10], "score": s.get("overall_score"),
                  "session_id": s.get("session_id", "")} for s in sess_sorted]

        avg = round(sum(scores) / len(scores), 1) if scores else None
        improvement = None
        if len(scores) >= 2:
            mid = len(scores) // 2
            first_half, second_half = scores[:mid] or scores, scores[mid:] or scores
            improvement = round((sum(second_half) / len(second_half)) - (sum(first_half) / len(first_half)), 1)

        profiles.append({
            "associate":          name,
            "is_unassigned":      name == UNASSIGNED,
            "sessions":           len(sess_sorted),
            "average_score":      avg,
            "latest_score":       scores[-1] if scores else None,
            "grade_distribution": grades,
            "improvement":        improvement,   # positive = trending up across history
            "trend":              trend,
            "last_session_date":  sess_sorted[-1].get("date", "") if sess_sorted else "",
        })

    named      = sorted([p for p in profiles if not p["is_unassigned"]], key=lambda p: -p["sessions"])
    unassigned = [p for p in profiles if p["is_unassigned"]]
    return named + unassigned


def _category_breakdown_for(session_ids: list[str]) -> dict[str, dict]:
    """
    Average category-level scores across sessions, using each session's
    full report from session_store. Sessions without a report on file
    (shouldn't normally happen post-migration, but be defensive) are
    silently skipped rather than failing — this is a best-effort breakdown.

    Uses get_reports_bulk() — ONE query for every session_id instead of
    one round trip per session. This used to be the dominant cost of
    both the dashboard and the history page's full=true response: against
    remote Postgres, 50+ sequential round trips is exactly what "stuck on
    Loading" looks like. Same result, just requested together.
    """
    reports = session_store.get_reports_bulk(session_ids)
    totals: dict[str, list[float]] = {}
    for sid in session_ids:
        report = reports.get(sid)
        if not report:
            continue
        cat_scores = (report.get("score") or {}).get("category_scores") or {}
        for cat_id, cs in cat_scores.items():
            if isinstance(cs, dict):
                raw, label = cs.get("raw_score"), cs.get("category", cat_id)
            else:
                raw, label = None, cat_id
            if isinstance(raw, (int, float)):
                totals.setdefault(label, []).append(raw)
    return {
        label: {"average": round(sum(vals) / len(vals), 1), "sessions": len(vals)}
        for label, vals in totals.items()
    }


def build_overall_dashboard(filters=None) -> dict:
    """
    All-audits rollup for the Audit Dashboard section — same aggregation
    approach as build_associate_profiles()/get_associate_detail(), just not
    grouped by associate. Purely additive: reads session_store, never
    touches the audit/scoring pipeline.

    `filters`: a history_filters.HistoryFilters, or None/empty for the
    unfiltered rollup — same shared filter object as Audit History/Export
    (see _load_sessions), so "Associate = Abdul" on the dashboard means the
    exact same session set as "Associate = Abdul" in the history list.

    Cached for _DASHBOARD_CACHE_TTL seconds, but ONLY the unfiltered call
    (filters is None or has no active filter) — every distinct filter
    combination would need its own cache slot otherwise, and a manager
    exploring different filter combos would mostly see the SAME stale
    result across different filters if this weren't scoped correctly.
    Filtered dashboards are cheap enough uncached at this data scale (see
    session_store.list_sessions_matching — one indexed query, no N+1).
    """
    filters_active = filters is not None and not filters.is_empty()
    if not filters_active:
        now = time.monotonic()
        if _dashboard_cache["data"] is not None and now < _dashboard_cache["expires_at"]:
            return _dashboard_cache["data"]
        result = _build_overall_dashboard_uncached(filters)
        _dashboard_cache["data"] = result
        _dashboard_cache["expires_at"] = now + _DASHBOARD_CACHE_TTL
        return result

    return _build_overall_dashboard_uncached(filters)


def _build_overall_dashboard_uncached(filters=None) -> dict:
    sessions = _load_sessions(filters)
    scores = [s.get("overall_score") for s in sessions if isinstance(s.get("overall_score"), (int, float))]
    session_ids = [s["session_id"] for s in sessions if s.get("session_id")]

    grades: dict[str, int] = {}
    for s in sessions:
        g = _grade_bucket(s.get("grade", ""))
        grades[g] = grades.get(g, 0) + 1

    sess_sorted = sorted(sessions, key=lambda s: s.get("date", ""))
    trend = [{"date": (s.get("date") or "")[:10], "score": s.get("overall_score"),
              "session_id": s.get("session_id", ""), "label": s.get("label", "")} for s in sess_sorted]

    recent_vs_previous = None
    sorted_scores = [t["score"] for t in trend if isinstance(t["score"], (int, float))]
    if len(sorted_scores) >= 2:
        half = max(len(sorted_scores) // 2, 1)
        previous, recent = sorted_scores[:-half] or sorted_scores, sorted_scores[-half:]
        recent_vs_previous = {
            "recent_avg": round(sum(recent) / len(recent), 1),
            "previous_avg": round(sum(previous) / len(previous), 1) if previous else None,
        }

    breakdown = _category_breakdown_for(session_ids)
    ranked = sorted(breakdown.items(), key=lambda kv: -kv[1]["average"])

    return {
        "total_audits":         len(sessions),
        "average_score":        round(sum(scores) / len(scores), 1) if scores else None,
        "grade_distribution":   grades,
        "category_breakdown":   breakdown,
        "strongest_categories": [k for k, _ in ranked[:3]],
        "weakest_categories":   [k for k, _ in ranked[-3:][::-1]] if len(ranked) > 3 else [],
        "trend":                trend,
        "recent_vs_previous":   recent_vs_previous,
    }


def get_associate_detail(name: str, filters=None) -> dict | None:
    match = next((p for p in build_associate_profiles(filters) if p["associate"] == name), None)
    if not match:
        return None
    session_ids = [t["session_id"] for t in match["trend"] if t.get("session_id")]
    breakdown = _category_breakdown_for(session_ids)
    ranked = sorted(breakdown.items(), key=lambda kv: -kv[1]["average"])
    return {
        **match,
        "category_breakdown":   breakdown,
        "strongest_categories": [k for k, _ in ranked[:3]],
        "weakest_categories":   [k for k, _ in ranked[-3:][::-1]] if len(ranked) > 3 else [],
    }
