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
import activity_log
import session_store

UNASSIGNED = "Unassigned"


def _is_real_label(label: str) -> bool:
    """A 'label' only counts as an associate name if it isn't blank and
    isn't the auto-generated fallback (a raw source URL)."""
    label = (label or "").strip()
    if not label:
        return False
    if label.lower().startswith(("http://", "https://")):
        return False
    return True


def _load_sessions() -> list[dict]:
    """Session summaries from session_store, remapped to the field names
    the rest of this module already expects (date/session_id/label/...) —
    with the label overridden by the tracker-push Lead Owner where one exists."""
    rows = session_store.list_sessions(limit=1000)
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


def build_associate_profiles() -> list[dict]:
    """One row per associate, named associates first (most sessions first),
    with anything unlabeled or URL-only grouped into a single 'Unassigned'
    row at the end."""
    sessions = _load_sessions()
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
    """
    totals: dict[str, list[float]] = {}
    for sid in session_ids:
        report = session_store.get_report(sid)
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


def build_overall_dashboard() -> dict:
    """
    All-audits rollup for the Audit Dashboard section — same aggregation
    approach as build_associate_profiles()/get_associate_detail(), just not
    grouped by associate. Purely additive: reads session_store, never
    touches the audit/scoring pipeline.
    """
    sessions = _load_sessions()
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


def get_associate_detail(name: str) -> dict | None:
    match = next((p for p in build_associate_profiles() if p["associate"] == name), None)
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
