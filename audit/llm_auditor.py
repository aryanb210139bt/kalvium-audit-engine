"""
audit/llm_auditor.py
Step 6c — Layer 2: LLM-based nuanced evaluation using OpenAI GPT-4o.
Evaluates qualities that rule-based systems can't assess:
  - Empathy and rapport
  - Objection handling quality
  - Storytelling and persuasion
  - Closing skills
  - Overall sales effectiveness
"""
from __future__ import annotations
import json
import logging
from typing import Any

from config.models import (
    Utterance, StructuredEvent, ScriptComplianceResult,
    TalkRatioResult, ObjectionAnalysis, LLMEvaluationResult,
)
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Evaluation categories with their prompts and weights
EVALUATION_CATEGORIES = [
    {
        "id": "product_explanation",
        "label": "Product explanation",
        "weight": 15,
        "prompt": (
            "Evaluate how clearly and compellingly the counsellor explained Kalvium's "
            "program. Did they cover unique value propositions? Did they explain it in "
            "simple, relatable terms? Did they adapt to the student/parent's level?"
        ),
    },
    {
        "id": "discovery_questions",
        "label": "Discovery questions",
        "weight": 15,
        "prompt": (
            "Evaluate the quality and quantity of discovery questions asked. "
            "Did the counsellor ask open-ended questions? Did they understand the "
            "student's goals, concerns, and background before pitching? "
            "Did they listen actively and build on answers?"
        ),
    },
    {
        "id": "engagement",
        "label": "Engagement",
        "weight": 15,
        "prompt": (
            "Evaluate overall engagement. Did the counsellor keep both student and parent "
            "involved? Did they use stories, examples, or data effectively? "
            "Was the energy level maintained? Did they avoid long monologues?"
        ),
    },
    {
        "id": "objection_handling",
        "label": "Objection handling",
        "weight": 15,
        "prompt": (
            "Evaluate how effectively the counsellor handled objections. "
            "Did they acknowledge concerns empathetically? Did they provide concrete "
            "data or testimonials? Were any objections dismissed or ignored? "
            "Was the resolution convincing?"
        ),
    },
    {
        "id": "parent_alignment",
        "label": "Parent alignment",
        "weight": 10,
        "prompt": (
            "Evaluate how well the counsellor addressed parent-specific concerns. "
            "Did they build trust with the parent? Did they address ROI for parents "
            "investing in fees? Did they speak to both student and parent, or only student?"
        ),
    },
    {
        "id": "confidence_clarity",
        "label": "Confidence and clarity",
        "weight": 10,
        "prompt": (
            "Evaluate the counsellor's communication quality: clarity of speech, "
            "confidence in answers, use of filler words, pacing, and overall "
            "professionalism. Did they sound knowledgeable and trustworthy?"
        ),
    },
    {
        "id": "closing_skills",
        "label": "Closing skills",
        "weight": 10,
        "prompt": (
            "Evaluate the closing technique. Did the counsellor create appropriate "
            "urgency? Did they ask for a commitment or next step? Did they give clear "
            "application instructions? Was the call ended with a clear CTA?"
        ),
    },
    {
        "id": "demo_completeness",
        "label": "Demo completeness",
        "weight": 10,
        "prompt": (
            "Evaluate whether the demo followed the required structure: intro, "
            "discovery, Kalvium explanation, repo/PPT demo, career outcomes, "
            "fee discussion, and closing. Were critical stages missing or rushed?"
        ),
    },
]


def _build_transcript_summary(
    utterances: list[Utterance],
    events: list[StructuredEvent],
    compliance: ScriptComplianceResult,
    talk_ratio: TalkRatioResult,
    objections: ObjectionAnalysis,
) -> str:
    """
    Build a compact, structured context for the LLM — not the raw transcript.
    This dramatically reduces token usage and improves evaluation consistency.
    """
    # Sample utterances (counsellor + key moments)
    counsellor_utts = [u for u in utterances if u.speaker.value == "Counsellor"][:20]
    student_utts    = [u for u in utterances if u.speaker.value == "Student"][:10]
    parent_utts     = [u for u in utterances if u.speaker.value == "Parent"][:10]

    def fmt_utts(utts, label):
        lines = [f"[{label}] {u.english_text[:150]}" for u in utts]
        return "\n".join(lines)

    # Structured events summary
    event_lines = [
        f"  {e.timestamp} | {e.speaker.value} | {e.event.value} | {e.sentiment.value}"
        for e in events[:40]
    ]

    objection_lines = [
        f"  - {o.category}: '{o.objection_text[:80]}' → {o.resolution_quality}"
        for o in objections.objections
    ]

    summary = f"""
=== DEMO CALL CONTEXT FOR AUDIT ===

SCRIPT COMPLETION: {compliance.completion_rate}%
Completed stages: {', '.join(k for k, v in compliance.model_dump().items() if v)}
Missing stages: {', '.join(k for k, v in compliance.model_dump().items() if not v)}

TALK RATIO:
  Counsellor: {talk_ratio.counsellor_pct}%
  Student: {talk_ratio.student_pct}%
  Parent: {talk_ratio.parent_pct}%
  Questions asked: {talk_ratio.total_questions_asked} (counsellor asked {talk_ratio.counsellor_questions})
  Interruptions: {talk_ratio.interruption_count}
  Dead air: {talk_ratio.dead_air_sec}s

OBJECTIONS ({objections.total_objections} total):
{chr(10).join(objection_lines) if objection_lines else "  None detected"}

KEY CONVERSATION EVENTS (chronological):
{chr(10).join(event_lines)}

COUNSELLOR UTTERANCES (sample):
{fmt_utts(counsellor_utts, "C")}

STUDENT UTTERANCES (sample):
{fmt_utts(student_utts, "S")}

PARENT UTTERANCES (sample):
{fmt_utts(parent_utts, "P")}
""".strip()
    return summary


class LLMAuditor:
    """
    Layer 2: Uses GPT-4o to evaluate nuanced aspects of the demo call.
    Returns structured scores, reasoning, and coaching suggestions.
    """

    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=settings.openai_api_key)
            except ImportError:
                raise RuntimeError("openai package not installed: pip install openai")
        return self._client

    def evaluate(
        self,
        utterances: list[Utterance],
        events: list[StructuredEvent],
        compliance: ScriptComplianceResult,
        talk_ratio: TalkRatioResult,
        objections: ObjectionAnalysis,
    ) -> list[LLMEvaluationResult]:
        """
        Run LLM evaluation for all categories.
        Falls back to rule-based scores if OpenAI key is not configured.
        """
        if not settings.openai_api_key:
            logger.warning("OPENAI_API_KEY not set — using rule-based fallback scores")
            return self._fallback_evaluate(compliance, talk_ratio, objections)

        context = _build_transcript_summary(
            utterances, events, compliance, talk_ratio, objections
        )

        results: list[LLMEvaluationResult] = []
        for category in EVALUATION_CATEGORIES:
            try:
                result = self._evaluate_category(category, context)
                results.append(result)
                logger.info(f"LLM eval [{category['label']}]: {result.score}/10")
            except Exception as exc:
                logger.error(f"LLM evaluation failed for {category['id']}: {exc}")
                results.append(self._default_result(category))

        return results

    def _evaluate_category(self, category: dict, context: str) -> LLMEvaluationResult:
        client = self._get_client()

        system_prompt = """You are an expert sales coach evaluating education counsellor performance.
You evaluate demo calls for Kalvium, a work-integrated tech education company targeting Indian students.
Return ONLY valid JSON. No markdown, no explanation outside the JSON."""

        user_prompt = f"""
Evaluate the following demo call on this criterion:

CRITERION: {category['label']}
EVALUATION FOCUS: {category['prompt']}

CALL CONTEXT:
{context}

Return a JSON object with exactly these fields:
{{
  "score": <float 0.0-10.0>,
  "reasoning": "<2-3 sentence explanation of the score>",
  "evidence": ["<specific quote or observation 1>", "<quote or observation 2>"],
  "suggestions": ["<specific coaching suggestion 1>", "<coaching suggestion 2>"]
}}

Score guidelines: 9-10=exceptional, 7-8=good, 5-6=average, 3-4=below average, 0-2=poor.
"""

        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=600,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content
        data: dict[str, Any] = json.loads(raw)

        return LLMEvaluationResult(
            category=category["label"],
            score=min(10.0, max(0.0, float(data.get("score", 5.0)))),
            reasoning=data.get("reasoning", ""),
            evidence=data.get("evidence", []),
            suggestions=data.get("suggestions", []),
        )

    # ── Fallback (no API key) ──────────────────────────────────────────────────

    def _fallback_evaluate(
        self,
        compliance: ScriptComplianceResult,
        talk_ratio: TalkRatioResult,
        objections: ObjectionAnalysis,
    ) -> list[LLMEvaluationResult]:
        """
        Deterministic fallback scores derived from rule-based analysis.
        Used when OpenAI key is not available.
        """
        scores: dict[str, tuple[float, str]] = {
            "product_explanation": (
                min(10, compliance.completion_rate / 10),
                "Estimated from script completion rate (no LLM).",
            ),
            "discovery_questions": (
                min(10, talk_ratio.counsellor_questions * 1.5),
                f"Counsellor asked {talk_ratio.counsellor_questions} questions.",
            ),
            "engagement": (
                8.0 if talk_ratio.is_balanced else 4.0,
                f"Talk balance: Counsellor {talk_ratio.counsellor_pct}% "
                f"(target <70%).",
            ),
            "objection_handling": (
                min(10, objections.handling_rate / 10),
                f"Handled {objections.resolved}/{objections.total_objections} objections well.",
            ),
            "parent_alignment": (
                6.0,
                "Fallback: manual review needed.",
            ),
            "confidence_clarity": (
                7.0,
                "Fallback: manual review needed.",
            ),
            "closing_skills": (
                6.0,
                "Fallback: manual review needed.",
            ),
            "demo_completeness": (
                compliance.completion_rate / 10,
                f"Script completion: {compliance.completion_rate}%.",
            ),
        }

        results: list[LLMEvaluationResult] = []
        for category in EVALUATION_CATEGORIES:
            score, reasoning = scores.get(category["id"], (5.0, "Fallback score."))
            results.append(LLMEvaluationResult(
                category=category["label"],
                score=round(score, 1),
                reasoning=f"[Rule-based fallback] {reasoning}",
                evidence=[],
                suggestions=["Set OPENAI_API_KEY for detailed coaching suggestions."],
            ))
        return results

    @staticmethod
    def _default_result(category: dict) -> LLMEvaluationResult:
        return LLMEvaluationResult(
            category=category["label"],
            score=5.0,
            reasoning="Evaluation unavailable due to API error.",
            evidence=[],
            suggestions=["Review this category manually."],
        )
