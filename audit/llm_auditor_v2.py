"""
audit/llm_auditor_v2.py
19-category LLM audit engine using GPT-4o.
Replaces llm_auditor.py's 8-category system.
"""
from __future__ import annotations
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from config.models import (
    Utterance, StructuredEvent, ScriptComplianceResult,
    TalkRatioResult, ObjectionAnalysis, LLMEvaluationResult,
)
from config.settings import get_settings
from audit.evaluation_framework import (
    EVALUATION_CATEGORIES, get_coaching, rule_score_category,
)

logger = logging.getLogger(__name__)
settings = get_settings()

AUDIT_WORKERS = 8


def _build_context_v2(
    utterances: list[Utterance],
    events: list[StructuredEvent],
    compliance: ScriptComplianceResult,
    talk_ratio: TalkRatioResult,
    objections: ObjectionAnalysis,
) -> str:
    """
    Stratified context: beginning + middle + end.
    Gives GPT-4o full visibility of the call arc.
    """
    n = len(utterances)
    start  = utterances[:12]
    mid_i  = max(12, n // 2 - 5)
    middle = utterances[mid_i: mid_i + 10]
    end    = utterances[max(0, n - 15):]

    seen, sampled = set(), []
    for u in start + middle + end:
        if u.utterance_id not in seen:
            sampled.append(u)
            seen.add(u.utterance_id)

    def fmt(utts):
        return "\n".join(
            f"[{u.speaker.value[:1]}|{int(u.start_time//60):02d}:{int(u.start_time%60):02d}] {u.english_text[:150]}"
            for u in utts
        )

    obj_lines = "\n".join(
        f"  [{o.category}] @{o.timestamp} → {o.resolution_quality}: '{o.objection_text[:80]}'"
        for o in objections.objections
    ) or "  None"

    script_done   = [k for k, v in compliance.model_dump().items() if v]
    script_missed = [k for k, v in compliance.model_dump().items() if not v]

    return f"""=== KALVIUM DEMO CALL — AUDIT CONTEXT ({n} utterances, sampled {len(sampled)}) ===

DETERMINISTIC METRICS (trust these, do not override):
  Script completion: {compliance.completion_rate:.0f}%
  Completed: {', '.join(script_done) or 'none'}
  Missed:    {', '.join(script_missed) or 'none'}
  Counsellor talk: {talk_ratio.counsellor_pct}% (target <65%)
  Student talk:    {talk_ratio.student_pct}%
  Parent talk:     {talk_ratio.parent_pct}%
  Counsellor questions: {talk_ratio.counsellor_questions} (target ≥5)
  Interruptions: {talk_ratio.interruption_count}
  Dead air: {talk_ratio.dead_air_sec}s
  Closing attempted: {compliance.closing_attempted}
  Fee discussed: {compliance.fee_structure_explained}

OBJECTIONS ({objections.total_objections} raised, {objections.resolved} resolved well):
{obj_lines}

TRANSCRIPT — OPENING (first 12 utterances):
{fmt(start)}

TRANSCRIPT — MIDDLE:
{fmt(middle)}

TRANSCRIPT — CLOSING (last 15 utterances):
{fmt(end)}

CONSISTENCY RULES (enforce strictly):
  - If counsellor_pct > 70%, two_way_communication and engagement cannot exceed 5/10
  - If closing_attempted=False, closing_skills cannot exceed 2/10
  - If counsellor_questions < 3, discovery_questions cannot exceed 4/10
  - If fee_structure_explained=False, fee_discussion cannot exceed 3/10
""".strip()


class LLMAuditorV2:
    """19-category GPT-4o audit engine."""

    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=settings.openai_api_key)
        return self._client

    def evaluate(
        self,
        utterances: list[Utterance],
        events: list[StructuredEvent],
        compliance: ScriptComplianceResult,
        talk_ratio: TalkRatioResult,
        objections: ObjectionAnalysis,
    ) -> list[LLMEvaluationResult]:
        if not settings.openai_api_key:
            logger.warning("No OPENAI_API_KEY — rule-based fallback for all 19 categories")
            return self._rule_fallback(utterances, compliance, talk_ratio, objections)

        context = _build_context_v2(utterances, events, compliance, talk_ratio, objections)
        results: list[LLMEvaluationResult | None] = [None] * len(EVALUATION_CATEGORIES)

        with ThreadPoolExecutor(max_workers=AUDIT_WORKERS) as ex:
            future_map = {
                ex.submit(self._evaluate_category, cat, context, utterances): (i, cat)
                for i, cat in enumerate(EVALUATION_CATEGORIES)
            }
            for future in as_completed(future_map):
                i, cat = future_map[future]
                try:
                    results[i] = future.result()
                    logger.info(f"[{cat['label']}]: {results[i].score}/10")
                except Exception as exc:
                    logger.error(f"[{cat['label']}] failed: {exc}")
                    results[i] = self._default_result(cat, utterances)

        return [r for r in results if r is not None]

    def _evaluate_category(
        self, category: dict, context: str, utterances: list[Utterance]
    ) -> LLMEvaluationResult:
        # Rule pre-score as anchor (prevents hallucinated extremes)
        rule_score, rule_evidence = rule_score_category(category["id"], utterances)

        client = self._get_client()
        user_prompt = f"""Evaluate this Kalvium counsellor demo call on ONE criterion.

CRITERION: {category['label']} (weight: {category['weight']}% of total score)
WHAT TO EVALUATE: {category['gpt_prompt']}

RULE-BASED PRE-SCORE: {rule_score}/10 (use as a sanity anchor — your score should be within ±3 of this unless you have strong transcript evidence)

CALL CONTEXT:
{context}

Return ONLY valid JSON:
{{
  "score": <float 0.0-10.0, one decimal>,
  "reasoning": "<3-4 sentences explaining the score with specific evidence from transcript>",
  "evidence": ["<specific quote or observation>", "<another evidence point>", "<third point>"],
  "missed_opportunities": ["<thing that was not done but should have been>"],
  "suggestions": ["<specific, actionable coaching suggestion 1>", "<suggestion 2>"]
}}

Scoring guide: 9-10=exceptional, 7-8=strong, 5-6=adequate, 3-4=weak, 0-2=absent/poor.
Base your score on TRANSCRIPT EVIDENCE, not assumptions."""

        response = self._get_client().chat.completions.create(
            model=settings.openai_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior sales coach evaluating education counsellors for Kalvium, "
                        "a work-integrated CS degree in India. Be evidence-based, specific, and fair. "
                        "Return ONLY valid JSON — no markdown, no explanation outside JSON."
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=500,
            response_format={"type": "json_object"},
        )

        data: dict[str, Any] = json.loads(response.choices[0].message.content)
        score = round(min(10.0, max(0.0, float(data.get("score", 5.0)))), 1)

        # Combine GPT evidence with rule-based evidence
        evidence = data.get("evidence", []) + rule_evidence
        missed   = data.get("missed_opportunities", [])
        coaching = get_coaching(category["id"], score)

        suggestions = data.get("suggestions", [])
        if coaching and coaching not in suggestions:
            suggestions.insert(0, coaching)

        return LLMEvaluationResult(
            category=category["label"],
            score=score,
            reasoning=data.get("reasoning", ""),
            evidence=evidence[:4],
            suggestions=suggestions[:3],
        )

    def _rule_fallback(
        self,
        utterances: list[Utterance],
        compliance: ScriptComplianceResult,
        talk_ratio: TalkRatioResult,
        objections: ObjectionAnalysis,
    ) -> list[LLMEvaluationResult]:
        """Pure rule-based scores when no OpenAI key is available."""
        results = []
        for cat in EVALUATION_CATEGORIES:
            score, evidence = rule_score_category(cat["id"], utterances)

            # Apply deterministic caps
            cat_id = cat["id"]
            if cat_id == "closing_skills" and not compliance.closing_attempted:
                score = min(score, 2.0)
            if cat_id == "fee_discussion" and not compliance.fee_structure_explained:
                score = min(score, 3.0)
            if cat_id == "discovery_questions" and talk_ratio.counsellor_questions < 3:
                score = min(score, 4.0)
            if cat_id == "two_way_communication" and talk_ratio.counsellor_pct > 70:
                score = min(score, 5.0)
            if cat_id == "objection_handling" and objections.missed > 0:
                score = max(0, score - objections.missed * 0.8)

            coaching = get_coaching(cat_id, score)
            results.append(LLMEvaluationResult(
                category=cat["label"],
                score=round(score, 1),
                reasoning=f"[Rule-based] Score derived from keyword analysis. Set OPENAI_API_KEY for detailed evaluation.",
                evidence=evidence,
                suggestions=[coaching] if coaching else ["Set OPENAI_API_KEY for detailed coaching."],
            ))
        return results

    @staticmethod
    def _default_result(cat: dict, utterances: list[Utterance]) -> LLMEvaluationResult:
        score, evidence = rule_score_category(cat["id"], utterances)
        coaching = get_coaching(cat["id"], score)
        return LLMEvaluationResult(
            category=cat["label"],
            score=round(score, 1),
            reasoning="LLM evaluation unavailable — using rule-based score.",
            evidence=evidence,
            suggestions=[coaching] if coaching else [],
        )
