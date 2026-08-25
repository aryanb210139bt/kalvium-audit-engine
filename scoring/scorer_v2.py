"""
scoring/scorer_v2.py
10-category weighted scoring engine + coaching generator.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Optional

from config.models import (
    ScriptComplianceResult, TalkRatioResult, IntentResult,
    ObjectionAnalysis, ExplainableEvaluationResult, CategoryScore,
)
from audit.evaluation_framework import EVALUATION_CATEGORIES, CATEGORY_WEIGHTS, get_coaching

logger = logging.getLogger(__name__)


def _grade(score: float) -> str:
    if score >= 90: return "A+"
    if score >= 80: return "A"
    if score >= 70: return "B"
    if score >= 60: return "C"
    if score >= 50: return "D"
    return "F"


@dataclass
class AuditScoreV2:
    overall: float
    grade: str
    category_scores: dict[str, CategoryScore] = field(default_factory=dict)

    # Derived metrics
    strongest_categories: list[str] = field(default_factory=list)
    weakest_categories: list[str] = field(default_factory=list)
    admission_probability: float = 0.0


class ScoringEngineV2:
    """
    Converts 10 LLM category scores into a final weighted score.
    Applies deterministic rule caps on top of LLM scores.
    """

    # LLM label → framework ID
    LABEL_TO_ID: dict[str, str] = {
        cat["label"]: cat["id"] for cat in EVALUATION_CATEGORIES
    }

    def compute_score(
        self,
        compliance: ScriptComplianceResult,
        talk_ratio: TalkRatioResult,
        intent: IntentResult,
        objections: ObjectionAnalysis,
        llm_results: list[ExplainableEvaluationResult],
    ) -> AuditScoreV2:
        # Map explainable results by category_id (direct — no label→ID lookup needed)
        llm_map: dict[str, ExplainableEvaluationResult] = {r.category_id: r for r in llm_results}

        category_scores: dict[str, CategoryScore] = {}

        for cat in EVALUATION_CATEGORIES:
            cat_id = cat["id"]
            llm = llm_map.get(cat_id)
            raw = llm.score if llm else 5.0

            # ── Deterministic rule adjustments ──────────────────────────────
            raw = self._apply_rules(cat_id, raw, compliance, talk_ratio, objections, intent)
            raw = round(max(0.0, min(10.0, raw)), 1)
            weight = CATEGORY_WEIGHTS[cat_id]
            weighted = round((raw / 10.0) * weight, 3)

            # Use why_score as the breakdown text (more informative than reasoning)
            breakdown = (llm.why_score or llm.coaching_feedback) if llm else "Rule-based estimate."

            category_scores[cat_id] = CategoryScore(
                category=cat["label"],
                weight=weight,
                raw_score=raw,
                weighted_score=weighted,
                breakdown=breakdown,
            )

        overall = round(sum(cs.weighted_score for cs in category_scores.values()), 1)
        overall = min(100.0, max(0.0, overall))

        # Rank categories
        sorted_cats = sorted(category_scores.values(), key=lambda c: c.raw_score, reverse=True)
        strongest = [c.category for c in sorted_cats[:3] if c.raw_score >= 6]
        weakest   = [c.category for c in sorted_cats[-3:] if c.raw_score < 6]

        # Admission probability
        prob = self._admission_probability(overall, intent, compliance, objections)

        logger.info(f"Score V2: {overall}/100 ({_grade(overall)})")
        return AuditScoreV2(
            overall=overall,
            grade=_grade(overall),
            category_scores=category_scores,
            strongest_categories=strongest,
            weakest_categories=weakest,
            admission_probability=prob,
        )

    def _apply_rules(
        self,
        cat_id: str,
        raw: float,
        compliance: ScriptComplianceResult,
        talk_ratio: TalkRatioResult,
        objections: ObjectionAnalysis,
        intent: IntentResult,
    ) -> float:
        """
        Deterministic rule adjustments for the 10-category framework.
        Calibrated generously for Indian education sales context.
        """
        if cat_id == "closing_skills":
            # KNET is the CTA — if no closing attempt at all, cap hard
            if not compliance.closing_attempted:
                raw = min(raw, 2.0)
            # Note: fee_structure_explained penalty removed — fee is its own category now

        elif cat_id == "product_explanation":
            # Blend script completion rate (40%) with LLM quality score (60%)
            # Generous: completion_rate is 0-100, scale to 0-10, weighted lightly
            rule_contribution = (compliance.completion_rate / 100.0) * 10.0
            raw = rule_contribution * 0.4 + raw * 0.6

        elif cat_id == "fee_discussion":
            if not compliance.fee_structure_explained:
                raw = min(raw, 4.0)  # softer cap: 4 instead of 3 (Indian context)

        elif cat_id == "discovery_questions":
            if talk_ratio.counsellor_questions >= 6:
                raw = min(10.0, raw + 0.5)
            elif talk_ratio.counsellor_questions <= 1:
                raw = max(0.0, raw - 1.5)  # softer penalty (was -2.0)

        elif cat_id == "two_way_communication":
            # Indian counsellors naturally speak more — only hard cap at extreme monologue
            if talk_ratio.counsellor_pct > 85:
                raw = min(raw, 4.0)
            elif talk_ratio.counsellor_pct > 75:
                raw = min(raw, 6.0)
            if talk_ratio.interruption_count > 12:
                raw = max(0, raw - 0.5)

        elif cat_id == "placement_credibility":
            if not compliance.placements_discussed:
                raw = min(raw, 4.0)

        return raw

    @staticmethod
    def _admission_probability(
        score: float,
        intent: IntentResult,
        compliance: ScriptComplianceResult,
        objections: ObjectionAnalysis,
    ) -> float:
        prob = 0.3  # base
        prob += score / 100 * 0.4  # audit score contribution

        if intent.student_intent.value in ("high_interest", "student_excited"):
            prob += 0.15
        if intent.parent_intent.value == "parent_resistance":
            prob -= 0.15
        if intent.alignment_risk == "high":
            prob -= 0.10
        if compliance.closing_attempted:
            prob += 0.05
        if objections.missed > 2:
            prob -= 0.10

        return round(max(0.0, min(1.0, prob)), 2)


# ─────────────────────────────────────────────────────────────────────────────
# Coaching generator V2
# ─────────────────────────────────────────────────────────────────────────────

class CoachingGeneratorV2:

    def generate(
        self,
        score: AuditScoreV2,
        compliance: ScriptComplianceResult,
        talk_ratio: TalkRatioResult,
        objections: ObjectionAnalysis,
        llm_results: list[ExplainableEvaluationResult],
    ) -> tuple[list[str], list[str], list[str], dict[str, list[str]]]:
        """
        Returns:
            (highlights, strengths, improvements, category_coaching)
        """
        highlights: list[str] = []
        strengths: list[str] = []
        improvements: list[str] = []
        category_coaching: dict[str, list[str]] = {}

        # Per-category coaching — prefer explainable result's coaching_feedback
        for cat_id, cs in score.category_scores.items():
            framework_coaching = get_coaching(cat_id, cs.raw_score)
            if framework_coaching:
                category_coaching[cs.category] = [framework_coaching]

        # Merge explainable coaching_feedback into category_coaching
        for r in llm_results:
            if r.coaching_feedback:
                existing = category_coaching.get(r.category, [])
                if r.coaching_feedback not in existing:
                    existing.insert(0, r.coaching_feedback)
                category_coaching[r.category] = existing

        # Talk ratio — generous threshold for Indian context
        if talk_ratio.counsellor_pct > 82:
            improvements.append(
                f"Talk dominance: You spoke {talk_ratio.counsellor_pct}% of the call — "
                "try to pause every 2 minutes and invite a response: "
                "'Does that make sense?' or 'What are your thoughts on this?'"
            )
        elif talk_ratio.counsellor_pct < 60:
            strengths.append(
                f"Great conversational balance — student/parent spoke "
                f"{100 - talk_ratio.counsellor_pct:.0f}% of the time."
            )

        # Questions
        if talk_ratio.counsellor_questions >= 5:
            strengths.append(
                f"Strong discovery — {talk_ratio.counsellor_questions} questions asked "
                "to understand the student's background and goals."
            )
        elif talk_ratio.counsellor_questions <= 2:
            improvements.append(
                f"Only {talk_ratio.counsellor_questions} discovery question(s) asked. "
                "Before pitching, ask at least 3-4 questions: stream, dream company, "
                "biggest concern about college — these answers shape the entire demo."
            )

        # KNET closing
        if compliance.closing_attempted:
            strengths.append(
                "Closing was attempted — great commitment to driving the next step. "
                "Make sure the KNET registration link is always the specific ask."
            )
        else:
            improvements.append(
                "No KNET closing attempt was made — this is the most important gap to fix. "
                "Every demo must end with: 'Shall I send you the KNET registration link right now?' "
                "KNET is how demos convert into admissions."
            )

        # Score highlights
        overall = score.overall
        if overall >= 85:
            highlights.append(f"Outstanding demo — {overall}/100 ({score.grade}). Top-tier counsellor performance.")
        elif overall >= 70:
            highlights.append(f"Strong demo — {overall}/100 ({score.grade}). A few targeted improvements will push this to A-grade.")
        elif overall >= 55:
            highlights.append(f"Developing — {overall}/100 ({score.grade}). Focus on the bottom 3 categories to add 15+ points.")
        else:
            highlights.append(
                f"Needs significant work — {overall}/100 ({score.grade}). "
                "Review closing, discovery, and objection handling immediately."
            )

        if score.strongest_categories:
            strengths.append(f"Highest scores: {', '.join(score.strongest_categories)}.")
        if score.weakest_categories:
            improvements.append(f"Priority improvement areas: {', '.join(score.weakest_categories)}.")

        # Top coaching from lowest-scoring categories
        llm_suggestions = [
            f"[{r.category}] {r.coaching_feedback}"
            for r in sorted(llm_results, key=lambda r: r.score)[:4]
            if r.coaching_feedback and "OPENAI" not in r.coaching_feedback
        ]
        improvements.extend(llm_suggestions)

        return (
            highlights[:3],
            list(dict.fromkeys(strengths))[:5],
            list(dict.fromkeys(improvements))[:8],
            category_coaching,
        )
