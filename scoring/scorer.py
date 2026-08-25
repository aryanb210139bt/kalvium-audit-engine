"""
scoring/scorer.py
Step 6d — Final weighted scoring engine (Layer 3 synthesis).
Combines deterministic rules, LLM scores, and ML signals into
the final AuditScore and generates coaching output.
"""
from __future__ import annotations
import logging
from config.models import (
    ScriptComplianceResult, TalkRatioResult, IntentResult,
    ObjectionAnalysis, LLMEvaluationResult,
    AuditScore, CategoryScore,
)

logger = logging.getLogger(__name__)

# ── Weight table (must sum to 100) ────────────────────────────────────────────
WEIGHTS: dict[str, float] = {
    "product_explanation":  15,
    "discovery_questions":  15,
    "engagement":           15,
    "objection_handling":   15,
    "parent_alignment":     10,
    "confidence_clarity":   10,
    "closing_skills":       10,
    "demo_completeness":    10,
}

# LLM category label → internal key mapping
LLM_LABEL_TO_KEY: dict[str, str] = {
    "Product explanation":   "product_explanation",
    "Discovery questions":   "discovery_questions",
    "Engagement":            "engagement",
    "Objection handling":    "objection_handling",
    "Parent alignment":      "parent_alignment",
    "Confidence and clarity":"confidence_clarity",
    "Closing skills":        "closing_skills",
    "Demo completeness":     "demo_completeness",
}


def _grade(score: float) -> str:
    if score >= 90: return "A+"
    if score >= 80: return "A"
    if score >= 70: return "B"
    if score >= 60: return "C"
    if score >= 50: return "D"
    return "F"


class ScoringEngine:
    """
    Combines all analysis layers into a final weighted score.

    Score = Σ (category_raw_score/10 × category_weight)
    where category scores come from:
      - LLM evaluation (primary for nuanced categories)
      - Rule-based adjustments (compliance, talk ratio)
    """

    def compute_score(
        self,
        compliance: ScriptComplianceResult,
        talk_ratio: TalkRatioResult,
        intent: IntentResult,
        objections: ObjectionAnalysis,
        llm_results: list[LLMEvaluationResult],
    ) -> AuditScore:
        # Map LLM results to keys
        llm_scores: dict[str, LLMEvaluationResult] = {
            LLM_LABEL_TO_KEY.get(r.category, r.category.lower().replace(" ", "_")): r
            for r in llm_results
        }

        # Build category scores with rule-based adjustments
        categories = {
            "product_explanation":  self._product_explanation(llm_scores, compliance),
            "discovery_questions":  self._discovery_questions(llm_scores, talk_ratio),
            "engagement":           self._engagement(llm_scores, talk_ratio),
            "objection_handling":   self._objection_handling(llm_scores, objections),
            "parent_alignment":     self._parent_alignment(llm_scores, intent),
            "confidence_clarity":   self._confidence_clarity(llm_scores),
            "closing_skills":       self._closing_skills(llm_scores, compliance),
            "demo_completeness":    self._demo_completeness(llm_scores, compliance),
        }

        # Weighted total (each raw_score is 0-10, weight is percentage)
        overall = sum(
            (cs.raw_score / 10) * cs.weight
            for cs in categories.values()
        )
        overall = round(min(100, max(0, overall)), 1)

        logger.info(f"Final audit score: {overall}/100 ({_grade(overall)})")
        for key, cs in categories.items():
            logger.debug(f"  {cs.category}: {cs.raw_score}/10 × {cs.weight}% = {cs.weighted_score}")

        return AuditScore(
            overall=overall,
            grade=_grade(overall),
            product_explanation=categories["product_explanation"],
            discovery_questions=categories["discovery_questions"],
            engagement=categories["engagement"],
            objection_handling=categories["objection_handling"],
            parent_alignment=categories["parent_alignment"],
            confidence_clarity=categories["confidence_clarity"],
            closing_skills=categories["closing_skills"],
            demo_completeness=categories["demo_completeness"],
        )

    # ── Category computers ────────────────────────────────────────────────────

    def _product_explanation(
        self, llm: dict, compliance: ScriptComplianceResult
    ) -> CategoryScore:
        base = llm.get("product_explanation")
        raw = base.score if base else 5.0

        # Rule adjustment: penalise if key topics not covered
        missing_count = sum([
            not compliance.kalvium_explanation,
            not compliance.career_outcomes,
            not compliance.traditional_comparison,
        ])
        raw = max(0.0, raw - missing_count * 0.5)

        return self._make_score("Product explanation", "product_explanation", raw, base)

    def _discovery_questions(
        self, llm: dict, talk_ratio: TalkRatioResult
    ) -> CategoryScore:
        base = llm.get("discovery_questions")
        raw = base.score if base else 5.0

        # Bonus for asking many questions
        if talk_ratio.counsellor_questions >= 5:
            raw = min(10, raw + 0.5)
        elif talk_ratio.counsellor_questions <= 1:
            raw = max(0, raw - 1.5)

        return self._make_score("Discovery questions", "discovery_questions", raw, base)

    def _engagement(
        self, llm: dict, talk_ratio: TalkRatioResult
    ) -> CategoryScore:
        base = llm.get("engagement")
        raw = base.score if base else 5.0

        # Hard penalty if counsellor monologues
        if talk_ratio.counsellor_pct > 80:
            raw = max(0, raw - 2.0)
        elif talk_ratio.counsellor_pct > 70:
            raw = max(0, raw - 1.0)

        # Penalty for excessive interruptions
        if talk_ratio.interruption_count > 10:
            raw = max(0, raw - 0.5)

        return self._make_score("Engagement", "engagement", raw, base)

    def _objection_handling(
        self, llm: dict, objections: ObjectionAnalysis
    ) -> CategoryScore:
        base = llm.get("objection_handling")
        raw = base.score if base else 5.0

        # Rule: adjust based on handling rate
        if objections.total_objections > 0:
            if objections.missed > 0:
                raw = max(0, raw - objections.missed * 0.8)
            if objections.resolved == objections.total_objections:
                raw = min(10, raw + 0.5)  # bonus for perfect handling

        return self._make_score("Objection handling", "objection_handling", raw, base)

    def _parent_alignment(
        self, llm: dict, intent: IntentResult
    ) -> CategoryScore:
        base = llm.get("parent_alignment")
        raw = base.score if base else 5.0

        # Penalise high alignment risk
        if intent.alignment_risk == "high":
            raw = max(0, raw - 2.0)
        elif intent.alignment_risk == "medium":
            raw = max(0, raw - 0.75)

        return self._make_score("Parent alignment", "parent_alignment", raw, base)

    def _confidence_clarity(self, llm: dict) -> CategoryScore:
        base = llm.get("confidence_clarity")
        raw = base.score if base else 6.0
        return self._make_score("Confidence and clarity", "confidence_clarity", raw, base)

    def _closing_skills(
        self, llm: dict, compliance: ScriptComplianceResult
    ) -> CategoryScore:
        base = llm.get("closing_skills")
        raw = base.score if base else 5.0

        if not compliance.closing_attempted:
            raw = max(0, raw - 2.5)   # Heavy penalty for not closing
        if not compliance.fee_structure_explained:
            raw = max(0, raw - 1.0)

        return self._make_score("Closing skills", "closing_skills", raw, base)

    def _demo_completeness(
        self, llm: dict, compliance: ScriptComplianceResult
    ) -> CategoryScore:
        # Demo completeness is 80% rule-based (completion rate), 20% LLM quality
        base = llm.get("demo_completeness")
        rule_score = compliance.completion_rate / 10
        llm_score  = base.score if base else rule_score
        raw = rule_score * 0.80 + llm_score * 0.20
        return self._make_score("Demo completeness", "demo_completeness", raw, base)

    # ── Helper ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _make_score(
        label: str, key: str, raw: float, llm: LLMEvaluationResult | None
    ) -> CategoryScore:
        raw = round(max(0.0, min(10.0, raw)), 1)
        weight = WEIGHTS[key]
        weighted = round((raw / 10) * weight, 2)
        breakdown = llm.reasoning if llm else "Rule-based score."
        return CategoryScore(
            category=label,
            weight=weight,
            raw_score=raw,
            weighted_score=weighted,
            breakdown=breakdown,
        )


# ── Coaching generator ─────────────────────────────────────────────────────────

class CoachingGenerator:
    """Generates human-readable coaching feedback from audit results."""

    def generate(
        self,
        score: AuditScore,
        compliance: ScriptComplianceResult,
        talk_ratio: TalkRatioResult,
        objections: ObjectionAnalysis,
        llm_results: list[LLMEvaluationResult],
    ) -> tuple[list[str], list[str], list[str]]:
        """
        Returns:
            (coaching_highlights, top_strengths, improvement_areas)
        """
        highlights: list[str] = []
        strengths:  list[str] = []
        improvements: list[str] = []

        # Talk ratio feedback
        if talk_ratio.counsellor_pct > 75:
            improvements.append(
                f"Talk time: You spoke {talk_ratio.counsellor_pct}% of the time — "
                "aim for under 65%. Ask more open-ended questions."
            )
        elif talk_ratio.counsellor_pct < 60:
            strengths.append("Excellent talk balance — student/parent felt heard.")

        # Questions
        if talk_ratio.counsellor_questions >= 5:
            strengths.append(
                f"Asked {talk_ratio.counsellor_questions} discovery questions — great job uncovering needs."
            )
        elif talk_ratio.counsellor_questions <= 2:
            improvements.append(
                f"Only {talk_ratio.counsellor_questions} discovery question(s) asked — "
                "aim for at least 4-5 before pitching."
            )

        # Script compliance
        missing = [k.replace("_", " ") for k, v in compliance.model_dump().items() if not v]
        if missing:
            improvements.append(f"Missed demo stages: {', '.join(missing[:3])}.")
        if compliance.closing_attempted:
            strengths.append("Attempted a closing — good follow-through.")
        else:
            improvements.append("No closing attempt detected — always ask for the next step.")

        # Objections
        if objections.missed > 0:
            improvements.append(
                f"{objections.missed} objection(s) left unaddressed: "
                + ", ".join(o.category for o in objections.objections if o.resolution_quality == "missed")[:3]
            )
        if objections.resolved > 0:
            strengths.append(
                f"Resolved {objections.resolved}/{objections.total_objections} objections effectively."
            )

        # LLM suggestions
        for result in llm_results:
            for suggestion in result.suggestions[:1]:  # top suggestion per category
                if suggestion and "fallback" not in suggestion.lower():
                    improvements.append(f"[{result.category}] {suggestion}")

        # Score-level highlights
        if score.overall >= 80:
            highlights.append(f"Strong demo — overall score {score.overall}/100 ({score.grade}).")
        elif score.overall >= 65:
            highlights.append(f"Good demo with areas to improve — {score.overall}/100 ({score.grade}).")
        else:
            highlights.append(
                f"Needs significant improvement — {score.overall}/100 ({score.grade}). "
                "Focus on talk balance and closing."
            )

        # Top performers
        categories = [
            score.product_explanation, score.discovery_questions, score.engagement,
            score.objection_handling, score.parent_alignment, score.confidence_clarity,
            score.closing_skills, score.demo_completeness,
        ]
        top = sorted(categories, key=lambda c: c.raw_score, reverse=True)[:2]
        for cat in top:
            if cat.raw_score >= 7:
                strengths.append(f"Strong {cat.category.lower()} ({cat.raw_score}/10).")

        # Deduplicate and cap
        return (
            highlights[:3],
            list(dict.fromkeys(strengths))[:5],
            list(dict.fromkeys(improvements))[:6],
        )
