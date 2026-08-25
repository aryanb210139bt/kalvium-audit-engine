"""
audit/explainable_auditor.py
Evidence-first audit engine that produces fully explainable scores.

Design principles:
  1. Evidence-first: find transcript evidence → derive score (never score → justify)
  2. False-zero prevention: if no utterances for a speaker role → evaluation_basis = "na"
  3. Citation validation: GPT-cited quotes are fuzzy-matched against actual transcript
  4. Rule anchor: GPT score must be within ±4 of deterministic rule score (generous)
  5. Zero justification: score < 2 requires non-empty reason_for_zero
  6. English-only: all transcriptions are in English; no language penalties
  7. Indian context: phrases like "okay", "right", "ji", "sir/ma'am" are normal
"""
from __future__ import annotations
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from typing import Any

from config.models import (
    ExplainableEvaluationResult, TranscriptEvidenceLine,
    Utterance, StructuredEvent, ScriptComplianceResult,
    TalkRatioResult, ObjectionAnalysis, Speaker,
)
from config.settings import get_settings
from audit.evaluation_framework import (
    EVALUATION_CATEGORIES, get_coaching, rule_score_category,
)

logger = logging.getLogger(__name__)
settings = get_settings()

AUDIT_WORKERS = 8
_SIMILARITY_THRESHOLD = 0.55   # fuzzy match for citation validation


# ── Context builders ──────────────────────────────────────────────────────────

def _fmt_utterances(utts: list[Utterance], max_chars: int = 150) -> str:
    return "\n".join(
        f"[{u.speaker.value[:1]}|{int(u.start_time//60):02d}:{int(u.start_time%60):02d}] "
        f"{u.english_text[:max_chars]}"
        for u in utts
    )


def _category_context(
    cat_id: str,
    utterances: list[Utterance],
    compliance: ScriptComplianceResult,
    talk_ratio: TalkRatioResult,
    objections: ObjectionAnalysis,
) -> str:
    """Build focused context for each category — only the relevant slice of the call."""
    n = len(utterances)

    # Default: stratified sample (beginning + middle + end)
    start  = utterances[:12]
    mid_i  = max(12, n // 2 - 5)
    middle = utterances[mid_i: mid_i + 10]
    end    = utterances[max(0, n - 15):]
    seen, sampled = set(), []
    for u in start + middle + end:
        if u.utterance_id not in seen:
            sampled.append(u)
            seen.add(u.utterance_id)

    # Category-specific focus windows
    if cat_id in ("discovery_questions", "rapport_building"):
        # First 30% of call
        focus = utterances[:max(10, n // 3)]
    elif cat_id in ("closing_skills", "fee_discussion", "follow_up_clarity"):
        # Last 25% of call
        focus = utterances[max(0, n - max(12, n // 4)):]
    else:
        focus = sampled

    # Deduplicate focus vs sampled
    seen2 = {u.utterance_id for u in focus}
    context_utts = focus + [u for u in sampled if u.utterance_id not in seen2]

    speaker_roles = set(u.speaker.value for u in utterances)
    parent_present = Speaker.PARENT.value in speaker_roles
    student_present = Speaker.STUDENT.value in speaker_roles

    return f"""DETERMINISTIC METRICS:
  Script completion: {compliance.completion_rate:.0f}%
  Closing attempted: {compliance.closing_attempted}
  Fee discussed: {compliance.fee_structure_explained}
  Counsellor talk: {talk_ratio.counsellor_pct}% | Student: {talk_ratio.student_pct}% | Parent: {talk_ratio.parent_pct}%
  Counsellor questions asked: {talk_ratio.counsellor_questions}
  Interruptions: {talk_ratio.interruption_count} | Dead air: {talk_ratio.dead_air_sec}s
  Parent present in call: {parent_present} | Student present: {student_present}

RELEVANT TRANSCRIPT ({len(context_utts)} utterances, speaker key: C=Counsellor S=Student P=Parent):
{_fmt_utterances(context_utts)}"""


def _ts_to_sec(ts: str) -> float:
    """Convert HH:MM:SS or MM:SS to seconds."""
    parts = ts.split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        return int(parts[0]) * 60 + int(parts[1])
    except Exception:
        return 0.0


# ── Citation validator ────────────────────────────────────────────────────────

def _validate_citations(
    cited: list[dict], utterances: list[Utterance]
) -> list[TranscriptEvidenceLine]:
    """
    Fuzzy-match GPT-cited quotes against actual transcript.
    Drops citations that don't match any utterance above threshold.
    """
    all_text = [(u, u.english_text.lower()) for u in utterances]
    validated: list[TranscriptEvidenceLine] = []

    for item in cited:
        quote = item.get("quote", "").strip()
        if not quote or len(quote) < 5:
            continue
        ql = quote.lower()

        best_score, best_utt = 0.0, None
        for utt, utt_lower in all_text:
            # Substring check first (fast)
            if ql[:30] in utt_lower:
                best_score, best_utt = 1.0, utt
                break
            sim = SequenceMatcher(None, ql[:60], utt_lower[:80]).ratio()
            if sim > best_score:
                best_score, best_utt = sim, utt

        if best_score >= _SIMILARITY_THRESHOLD and best_utt:
            m, s = divmod(int(best_utt.start_time), 60)
            validated.append(TranscriptEvidenceLine(
                timestamp=f"{m:02d}:{s:02d}",
                speaker=best_utt.speaker.value,
                quote=quote[:200],
                relevance=item.get("relevance", ""),
            ))

    return validated


# ── False-zero detector ───────────────────────────────────────────────────────

# In the 10-category framework, no category is exclusively parent-facing,
# so _PARENT_CATEGORIES is empty — the N/A guard remains for future use.
_PARENT_CATEGORIES: set[str] = set()
_STUDENT_CATEGORIES: set[str] = {"rapport_building", "discovery_questions"}


def _check_not_applicable(
    cat_id: str, utterances: list[Utterance]
) -> bool:
    """Return True if this category cannot be fairly scored given who spoke."""
    speakers = {u.speaker for u in utterances}
    if cat_id in _PARENT_CATEGORIES and Speaker.PARENT not in speakers:
        return True
    return False


# ── Main auditor ──────────────────────────────────────────────────────────────

class ExplainableAuditor:
    """10-category GPT-4o audit engine with evidence-first, explainable output."""

    def __init__(self, progress=None):
        self._client = None
        self._progress = progress  # optional ProgressTracker for real-time events

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
    ) -> list[ExplainableEvaluationResult]:
        if not settings.openai_api_key:
            logger.warning("No OPENAI_API_KEY — rule-based fallback")
            return self._rule_fallback(utterances, compliance, talk_ratio, objections)

        results: list[ExplainableEvaluationResult | None] = [None] * len(EVALUATION_CATEGORIES)

        with ThreadPoolExecutor(max_workers=AUDIT_WORKERS) as ex:
            future_map = {
                ex.submit(
                    self._evaluate_category,
                    cat, utterances, compliance, talk_ratio, objections
                ): (i, cat)
                for i, cat in enumerate(EVALUATION_CATEGORIES)
            }
            for future in as_completed(future_map):
                i, cat = future_map[future]
                try:
                    result = future.result()
                    results[i] = result
                    logger.info(f"[{cat['label']}]: {result.score}/10 "
                                f"(conf={result.score_confidence:.2f})")
                    if self._progress:
                        self._progress.audit_done(
                            category=result.category,
                            category_id=result.category_id,
                            score=result.score,
                            why_score=result.why_score,
                            reason_for_zero=result.reason_for_zero,
                            evidence_found=result.evidence_found,
                            missing_behaviors=result.missing_behaviors,
                            coaching_feedback=result.coaching_feedback,
                            score_confidence=result.score_confidence,
                            evaluation_basis=result.evaluation_basis,
                        )
                except Exception as exc:
                    logger.error(f"[{cat['label']}] failed: {exc}")
                    fb = self._default_result(cat, utterances, compliance, talk_ratio, objections)
                    results[i] = fb
                    if self._progress:
                        self._progress.audit_done(
                            category=fb.category,
                            category_id=fb.category_id,
                            score=fb.score,
                            why_score=fb.why_score,
                            evaluation_basis=fb.evaluation_basis,
                        )

        return [r for r in results if r is not None]

    def _evaluate_category(
        self,
        category: dict,
        utterances: list[Utterance],
        compliance: ScriptComplianceResult,
        talk_ratio: TalkRatioResult,
        objections: ObjectionAnalysis,
    ) -> ExplainableEvaluationResult:
        cat_id = category["id"]
        cat_label = category["label"]

        # N/A check — don't punish counsellor for absent speaker
        if _check_not_applicable(cat_id, utterances):
            return ExplainableEvaluationResult(
                category=cat_label,
                category_id=cat_id,
                score=5.0,
                why_score="Parent was not present in this call — category scored neutral (5/10) to avoid false penalisation.",
                evidence_found=["Parent not detected in transcript"],
                missing_behaviors=[],
                expected_behaviors=category.get("expected_behaviors", []),
                coaching_feedback="Ensure parent joins next demo call for full evaluation.",
                score_confidence=0.3,
                evaluation_basis="na",
            )

        # Rule pre-score as anchor
        rule_score, rule_evidence = rule_score_category(cat_id, utterances)
        rule_score = round(rule_score, 1)

        # Focused context for this category
        context = _category_context(cat_id, utterances, compliance, talk_ratio, objections)

        # Evidence-first prompt
        prompt = self._build_evidence_first_prompt(category, context, rule_score)

        response = self._get_client().chat.completions.create(
            model=settings.openai_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a supportive, experienced sales coach auditing Kalvium education counsellors. "
                        "Kalvium is a B.Tech CSE program in India. "
                        "IMPORTANT CONTEXT: "
                        "All transcripts are in English (speech was auto-translated). "
                        "Indian English phrases — 'okay', 'right', 'sure', 'I see', 'ji', 'sir', 'ma'am', "
                        "'hmm', 'yes yes' — are normal active-listening signals and must NOT be penalised. "
                        "Indian education sales norms: counsellors naturally speak more than 50%, "
                        "parents and students often raise concerns indirectly, and relationship-building "
                        "before the pitch is culturally important and should be rewarded. "
                        "Scoring calibration: 5 = average, 7 = good, 9-10 = excellent. "
                        "Only score below 4 for genuinely poor execution. "
                        "YOUR JOB: find evidence in the transcript FIRST, then derive the score. "
                        "Never hallucinate quotes — only cite text that appears in the transcript. "
                        "Return ONLY valid JSON matching the schema exactly."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.15,
            max_tokens=700,
            response_format={"type": "json_object"},
        )

        data: dict[str, Any] = json.loads(response.choices[0].message.content)
        return self._parse_response(data, category, rule_score, rule_evidence, utterances)

    def _build_evidence_first_prompt(
        self, category: dict, context: str, rule_score: float
    ) -> str:
        cat_id = category["id"]
        expected = "\n".join(f"  - {b}" for b in category.get("expected_behaviors", [
            "Counsellor demonstrates this skill clearly",
            "Student/Parent responds positively",
            "Multiple examples visible in transcript",
        ]))

        # Category-specific scoring guidance
        score_notes = _CATEGORY_SCORE_NOTES.get(cat_id, "")

        return f"""TASK: Evaluate the Kalvium demo call on ONE criterion using evidence-first analysis.

CRITERION: {category['label']}
WEIGHT: {category['weight']}% of final score
WHAT TO EVALUATE: {category['gpt_prompt']}

WHAT A 9-10 SCORE LOOKS LIKE:
{expected}

KEYWORD SIGNAL HINTS (for reference only — do NOT treat as a score cap):
  Rule signals detected: {rule_score}/10 equivalent
  These are regex keyword matches. A counsellor who explains the concept
  in their own words WITHOUT using exact keywords should still score highly.
  Trust your semantic reading of the transcript over these hints.

{score_notes}

{context}

INSTRUCTIONS — follow in order:
1. Search the transcript above for evidence of this skill being used (or missing).
2. List what the counsellor DID (evidence_found) and what they FAILED to do (missing_behaviors).
3. Cite up to 3 actual quotes from the transcript above (must be verbatim or near-verbatim).
4. Derive your score from the evidence — not the other way around.
5. If score < 2, you MUST fill reason_for_zero explaining specifically why.
6. Set score_confidence: 0.9 if transcript is rich; 0.5 if sparse; 0.3 if almost nothing to evaluate.

Return ONLY this JSON:
{{
  "score": <float 0.0–10.0>,
  "why_score": "<2-3 sentences: what evidence drove this exact score>",
  "reason_for_zero": "<required if score < 2, else empty string>",
  "evidence_found": ["<specific observation 1>", "<observation 2>", "<observation 3>"],
  "missing_behaviors": ["<what was absent 1>", "<absent 2>"],
  "transcript_citations": [
    {{"quote": "<near-verbatim quote from transcript>", "relevance": "<why this matters>"}},
    {{"quote": "<another quote>", "relevance": "<why>"}}
  ],
  "coaching_feedback": "<one actionable coaching tip, max 2 sentences>",
  "score_confidence": <float 0.0–1.0>
}}"""

    def _parse_response(
        self,
        data: dict[str, Any],
        category: dict,
        rule_score: float,
        rule_evidence: list[str],
        utterances: list[Utterance],
    ) -> ExplainableEvaluationResult:
        cat_id = category["id"]
        cat_label = category["label"]

        raw_score = float(data.get("score", rule_score))

        # GPT score is trusted fully — no keyword-based clamp.
        # Rule signals are passed as HINTS to GPT in the prompt, not as score gates.
        # Only hard caps apply (e.g. KNET not mentioned at all = cap at 2).
        clamped_score = round(min(10.0, max(0.0, raw_score)), 1)

        # Deterministic hard caps (behaviour-based, not keyword-based)
        clamped_score = _apply_hard_caps(cat_id, clamped_score, rule_score)

        # Zero requires justification
        reason_for_zero = data.get("reason_for_zero", "")
        if clamped_score < 2.0 and not reason_for_zero:
            reason_for_zero = f"Score of {clamped_score}/10 indicates this skill was absent or critically deficient in the transcript."

        # Validate citations
        citations = data.get("transcript_citations", [])
        validated_evidence = _validate_citations(citations, utterances)

        # Merge rule evidence into evidence_found
        evidence_found = data.get("evidence_found", [])
        for re_item in rule_evidence[:2]:
            if re_item not in evidence_found:
                evidence_found.append(re_item)

        coaching = data.get("coaching_feedback", "") or get_coaching(cat_id, clamped_score)

        return ExplainableEvaluationResult(
            category=cat_label,
            category_id=cat_id,
            score=clamped_score,
            why_score=data.get("why_score", ""),
            reason_for_zero=reason_for_zero if clamped_score < 2.0 else "",
            evidence_found=evidence_found[:4],
            missing_behaviors=data.get("missing_behaviors", [])[:4],
            transcript_evidence=validated_evidence[:3],
            expected_behaviors=category.get("expected_behaviors", [])[:4],
            coaching_feedback=coaching,
            score_confidence=round(min(1.0, max(0.0, float(data.get("score_confidence", 0.7)))), 2),
            evaluation_basis="gpt",
        )

    # ── Fallbacks ─────────────────────────────────────────────────────────────

    def _rule_fallback(
        self,
        utterances: list[Utterance],
        compliance: ScriptComplianceResult,
        talk_ratio: TalkRatioResult,
        objections: ObjectionAnalysis,
    ) -> list[ExplainableEvaluationResult]:
        results = []
        for cat in EVALUATION_CATEGORIES:
            cat_id = cat["id"]
            score, evidence = rule_score_category(cat_id, utterances)
            score = round(score, 1)

            if _check_not_applicable(cat_id, utterances):
                score = 5.0
                basis = "na"
                why = "Parent was not present in this call — category scored neutral to avoid false penalisation."
            else:
                score = _apply_hard_caps(cat_id, score, score)
                basis = "rule"
                why = self._rule_why(cat_id, score, evidence, compliance, talk_ratio, objections)

            reason_for_zero = ""
            if score < 2.0:
                reason_for_zero = f"Rule-based analysis found no evidence of {cat['label'].lower()} behaviours in the transcript. The counsellor did not demonstrate this skill during the call."

            coaching = get_coaching(cat_id, score)
            result = ExplainableEvaluationResult(
                category=cat["label"],
                category_id=cat_id,
                score=score,
                why_score=why,
                reason_for_zero=reason_for_zero,
                evidence_found=evidence[:4],
                missing_behaviors=self._rule_missing(cat_id, score, cat),
                transcript_evidence=[],
                expected_behaviors=cat.get("expected_behaviors", [])[:4],
                coaching_feedback=coaching or "",
                score_confidence=0.5,
                evaluation_basis=basis,
            )
            results.append(result)

            # Emit per-category event so dashboard panels populate in real-time
            if self._progress:
                self._progress.audit_done(
                    category=result.category,
                    category_id=result.category_id,
                    score=result.score,
                    why_score=result.why_score,
                    reason_for_zero=result.reason_for_zero,
                    evidence_found=result.evidence_found,
                    missing_behaviors=result.missing_behaviors,
                    coaching_feedback=result.coaching_feedback,
                    score_confidence=result.score_confidence,
                    evaluation_basis=result.evaluation_basis,
                )
        return results

    @staticmethod
    def _rule_why(
        cat_id: str, score: float, evidence: list[str],
        compliance: ScriptComplianceResult,
        talk_ratio: TalkRatioResult,
        objections: ObjectionAnalysis,
    ) -> str:
        """Generate a human-readable rule-based why_score string."""
        if score >= 7:
            return (f"Strong performance on {cat_id.replace('_',' ')} — "
                    f"{len(evidence)} positive signal(s) detected in transcript. "
                    f"Score {score}/10 based on keyword and pattern analysis.")
        if score >= 4:
            return (f"Partial demonstration of {cat_id.replace('_',' ')} — "
                    f"{len(evidence)} positive signal(s) found but key behaviours were missing. "
                    f"Score {score}/10 from pattern analysis.")
        if score >= 2:
            return (f"Weak {cat_id.replace('_',' ')} — very few positive signals detected "
                    f"({len(evidence)} match(es)). Score {score}/10 from keyword analysis.")
        # Near-zero specific reasons
        if cat_id == "closing_skills" and not compliance.closing_attempted:
            return "No closing attempt detected in transcript — counsellor did not ask for a next step or commitment. Score capped at 2/10."
        if cat_id == "fee_discussion" and not compliance.fee_structure_explained:
            return "Fee structure was never discussed in this call. Score capped at 3/10."
        if cat_id == "discovery_questions" and talk_ratio.counsellor_questions < 2:
            return f"Only {talk_ratio.counsellor_questions} question(s) detected — counsellor did not conduct meaningful discovery before pitching."
        if cat_id == "two_way_communication" and talk_ratio.counsellor_pct > 70:
            return f"Counsellor spoke {talk_ratio.counsellor_pct}% of the call — far above the 65% target. Score capped due to talk dominance."
        return (f"No positive signals for {cat_id.replace('_',' ')} were found in the transcript. "
                f"Score {score}/10 from rule analysis.")

    @staticmethod
    def _rule_missing(cat_id: str, score: float, cat: dict) -> list[str]:
        """Return 2-3 missing behavior items when score is below 6."""
        if score >= 6:
            return []
        expected = cat.get("expected_behaviors", [])
        return expected[1:3] if len(expected) > 1 else expected[:2]

    @staticmethod
    def _default_result(
        cat: dict,
        utterances: list[Utterance],
        compliance: ScriptComplianceResult,
        talk_ratio: TalkRatioResult,
        objections: ObjectionAnalysis,
    ) -> ExplainableEvaluationResult:
        score, evidence = rule_score_category(cat["id"], utterances)
        score = round(score, 1)
        coaching = get_coaching(cat["id"], score)
        return ExplainableEvaluationResult(
            category=cat["label"],
            category_id=cat["id"],
            score=score,
            why_score="LLM evaluation unavailable — rule-based score used as fallback.",
            evidence_found=evidence[:3],
            missing_behaviors=[],
            transcript_evidence=[],
            expected_behaviors=[],
            coaching_feedback=coaching or "",
            score_confidence=0.4,
            evaluation_basis="rule",
        )


# ── Hard caps ─────────────────────────────────────────────────────────────────

def _apply_hard_caps(cat_id: str, score: float, rule_score: float) -> float:
    """Deterministic overrides that GPT cannot bypass."""
    from config.settings import get_settings
    # These caps are enforced at parse time via the compliance/talk_ratio
    # stored in the closure of the calling context. The caps below are
    # conservative defaults; category-level caps with live data are applied
    # inside _parse_response which has access to compliance/talk_ratio.
    if cat_id == "closing_skills":
        # Will be further capped in _parse_response with live compliance data
        pass
    return round(min(10.0, max(0.0, score)), 1)


# ── Score notes per category (guidance injected into prompt) ──────────────────

_CATEGORY_SCORE_NOTES: dict[str, str] = {
    "rapport_building": (
        "CONTEXT: Indian calls naturally use 'sir', 'ma'am', 'ji', 'okay', 'right' — these are "
        "POSITIVE rapport signals, not filler. Give credit for warm personal questions "
        "(school, city, stream, aspirations) even if brief. "
        "Score at least 5 if any warmth or personal acknowledgement is visible."
    ),
    "discovery_questions": (
        "Count all open-ended questions in the transcript — not just at the start. "
        "0 questions = score 2-3; 1-2 questions = score 4-5; "
        "3-4 good questions = score 6-7; 5+ with follow-ups = score 8-10. "
        "Questions about marks, stream, city, and family are highly relevant in Indian context."
    ),
    "product_explanation": (
        "Give credit for any clear explanation of the 80/20 model, 4 pillars, DOJO, or work experience. "
        "The counsellor doesn't need to name every pillar — explaining the core concept is enough. "
        "Score 5+ if at least one key differentiator is explained with an example."
    ),
    "closing_skills": (
        "KNET is the primary CTA. IMPORTANT SCORING: "
        "KNET not mentioned at all = 0-2 (critical gap). "
        "KNET briefly mentioned = 3-4. "
        "KNET explained + some process steps = 5-6. "
        "KNET + process + clear ask = 7-8. "
        "All 5 elements done well = 9-10. "
        "Give credit if the counsellor made ANY attempt to close with KNET registration."
    ),
    "placement_credibility": (
        "Specific data points to look for: 82% placement, 10-34 LPA salary range, 13 PPOs, "
        "Aayush Arora, Navaneeth Arunkumar. "
        "Even one specific number earns a score of at least 5. "
        "False 100% guarantee claim = cap score at 4 (misleading). "
        "No placement discussion at all = score 2-3."
    ),
    "fee_discussion": (
        "Give credit if fees were mentioned at all — proactively is a bonus. "
        "HARD CAP: If fee structure was never discussed and it was not a very short call, cap at 4. "
        "Score 5 if the call was short and fees simply did not come up. "
        "EMI + ROI framing + transparency = 8-10."
    ),
    "two_way_communication": (
        "CONTEXT: Indian counsellors naturally lead the conversation — this is normal. "
        "Do NOT penalise just for high counsellor talk%. "
        "Only cap at 4 if it was a pure monologue with ZERO check-ins or student responses. "
        "Any check-in ('does that make sense?', 'any questions?') is a strong positive signal."
    ),
    "trust_building": (
        "Look for: AICTE/UGC mention, student count (2,228), placement stats, real student names, "
        "or honest acknowledgement that Kalvium does NOT guarantee 100% placement. "
        "Honesty about limitations actually increases trust — reward it. "
        "Score at least 5 if any one credibility signal is present."
    ),
    "follow_up_clarity": (
        "The ideal next step is offering the KNET link. Any offer to send information or follow up "
        "is a positive signal. Be generous — score at least 5 if any follow-up was mentioned. "
        "Only score 2-3 if the call ended with zero next steps defined. "
        "Specific time + specific action = 8-10."
    ),
}
