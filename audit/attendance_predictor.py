"""
audit/attendance_predictor.py
Layer 5 — Webinar Attendance Prediction.

Predicts probability that the registered prospect will actually attend
the webinar, based on commitment language, joining instructions discussed,
and cross-layer behavioral signals.
"""
from __future__ import annotations
import json
import logging
import re

from openai import OpenAI

from config.models import Utterance, Speaker
from config.kalvium_models import (
    AttendancePredictionResult, KalviumComplianceResult, BuyerIntentResult
)
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Strong commitment language patterns
_STRONG_COMMIT = [
    r"\bpakka aaunga\b", r"\bpakka join\b", r"\bpakka attend\b",
    r"\bwill definitely\b", r"\bI will be there\b", r"\bI will join\b",
    r"\bconfirm.*attend\b", r"\byes.*join.*session\b", r"\bnote kar liya\b",
    r"\breminder.*set\b", r"\bcalendar.*mark\b", r"\bI promise\b",
]

# Weak / uncertain commitment language
_WEAK_COMMIT = [
    r"\bwill try\b", r"\btry karenge\b", r"\bmaybe\b", r"\bshayad\b",
    r"\bdekhte hain\b", r"\bdekh lenge\b", r"\bwill see\b",
    r"\bif possible\b", r"\bho payega\b", r"\bhoga toh\b",
    r"\blet's see\b", r"\bno promises\b",
]

# Joining-intent signals
_JOIN_SIGNALS = [
    r"\bteams.*link\b", r"\bjoin.*link\b", r"\bhow.*join\b",
    r"\blink.*send\b", r"\blink.*mail\b", r"\bcheck.*inbox\b",
    r"\bwhat.*time.*session\b", r"\bsession.*time\b", r"\bschedule\b",
    r"\bwhich.*day\b", r"\bkitne baje\b", r"\bkab hai\b",
]


def _rule_attendance_signals(utterances: list[Utterance], full_text: str) -> dict:
    strong_hits = sum(
        1 for p in _STRONG_COMMIT if re.search(p, full_text, re.IGNORECASE)
    )
    weak_hits = sum(
        1 for p in _WEAK_COMMIT if re.search(p, full_text, re.IGNORECASE)
    )
    join_hits = sum(
        1 for p in _JOIN_SIGNALS if re.search(p, full_text, re.IGNORECASE)
    )

    # Pull prospect commitment quotes
    commit_quotes = []
    for u in utterances:
        if u.speaker in (Speaker.STUDENT, Speaker.UNKNOWN):
            text = u.english_text
            if any(re.search(p, text, re.IGNORECASE)
                   for p in _STRONG_COMMIT + _WEAK_COMMIT):
                commit_quotes.append(text[:120])

    return {
        "strong_commit_hits": strong_hits,
        "weak_commit_hits": weak_hits,
        "join_signal_hits": join_hits,
        "commit_quotes": commit_quotes[:5],
    }


def _build_attendance_prompt(
    full_transcript: str,
    compliance: KalviumComplianceResult,
    intent: BuyerIntentResult,
) -> str:
    return f"""You are a predictive analytics specialist for ed-tech webinar attendance.

Research shows genuine webinar attendees:
- Confirm the date/time explicitly
- Ask how to join (Teams link, email)
- Mention they will check their schedule or tell parents
- Show interest in the actual content
- Confirm email receipt expectations
- Use definite language: "I will attend", "pakka aaunga"

Non-attendees typically:
- Vaguely agree without specific time confirmation
- Never ask joining instructions
- Show low energy throughout the call
- Use uncertain language: "will try", "maybe", "dekhte hain"
- Are passive throughout the call

COMPLIANCE DATA:
- Email collected: {compliance.email_collected}
- Email verified: {compliance.email_verified}
- Attendance commitment obtained: {compliance.attendance_commitment_obtained}
- Interest score: {intent.interest_score}
- Buying intent: {intent.buying_intent_score}

TRANSCRIPT:
{full_transcript[:8000]}

TASK: Predict probability of webinar attendance (0.0 = will not attend, 1.0 = certain to attend).

Return ONLY valid JSON:
{{
  "attendance_probability": <0.0-1.0>,
  "commitment_language_score": <1-10>,
  "commitment_quotes": ["exact quotes showing commitment or lack of it"],
  "joining_instructions_discussed": <true/false>,
  "time_confirmed": <true/false>,
  "schedule_conflict_mentioned": <true/false>,
  "follow_through_indicators": ["positive signals suggesting they will attend"],
  "risk_factors": ["factors suggesting they will not attend"],
  "prediction_confidence": "HIGH/MEDIUM/LOW",
  "prediction_reasoning": "<paragraph explaining prediction>"
}}"""


class AttendancePredictor:
    """Layer 5: Predict webinar attendance probability."""

    def __init__(self):
        if settings.openai_api_key:
            self._client = OpenAI(api_key=settings.openai_api_key)
        else:
            self._client = None

    def predict(
        self,
        utterances: list[Utterance],
        full_transcript: str,
        compliance: KalviumComplianceResult,
        intent: BuyerIntentResult,
        engagement_score: float,
        fake_booking_probability: float,
    ) -> AttendancePredictionResult:
        rule_signals = _rule_attendance_signals(utterances, full_transcript)

        # Short-circuit: high fake probability → very low attendance
        if fake_booking_probability > 0.70:
            return AttendancePredictionResult(
                attendance_probability=round(max(0.05, 0.30 - fake_booking_probability * 0.3), 3),
                commitment_language_score=1,
                commit_quotes=rule_signals["commit_quotes"],
                joining_instructions_discussed=bool(rule_signals["join_signal_hits"]),
                time_confirmed=False,
                schedule_conflict_mentioned=False,
                follow_through_indicators=[],
                risk_factors=["High fake booking probability — attendance extremely unlikely"],
                prediction_confidence="HIGH",
                prediction_reasoning=(
                    f"Fake booking probability is {fake_booking_probability:.0%}. "
                    "Attendance prediction is suppressed to near zero."
                ),
            )

        if not self._client:
            return self._rule_predict(rule_signals, compliance, intent, engagement_score)

        prompt = _build_attendance_prompt(full_transcript, compliance, intent)
        try:
            resp = self._client.chat.completions.create(
                model=settings.openai_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=700,
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content or "{}")
            return self._parse(data, rule_signals)
        except Exception as exc:
            logger.error(f"AttendancePredictor LLM call failed: {exc}")
            return self._rule_predict(rule_signals, compliance, intent, engagement_score)

    # ── parsers ───────────────────────────────────────────────────────────────

    def _parse(self, data: dict, rule_signals: dict) -> AttendancePredictionResult:
        ap = float(data.get("attendance_probability", 0.3))
        cls = int(data.get("commitment_language_score", 3))

        # Merge rule quotes with LLM quotes
        quotes = list(data.get("commitment_quotes", []))
        for q in rule_signals["commit_quotes"]:
            if q not in quotes:
                quotes.append(q)

        return AttendancePredictionResult(
            attendance_probability=round(max(0.0, min(1.0, ap)), 3),
            commitment_language_score=max(1, min(10, cls)),
            commitment_quotes=quotes[:5],
            joining_instructions_discussed=bool(data.get("joining_instructions_discussed", False))
                or bool(rule_signals["join_signal_hits"]),
            time_confirmed=bool(data.get("time_confirmed", False)),
            schedule_conflict_mentioned=bool(data.get("schedule_conflict_mentioned", False)),
            follow_through_indicators=data.get("follow_through_indicators", []),
            risk_factors=data.get("risk_factors", []),
            prediction_confidence=str(data.get("prediction_confidence", "LOW")),
            prediction_reasoning=str(data.get("prediction_reasoning", "")),
        )

    def _rule_predict(
        self,
        rule_signals: dict,
        compliance: KalviumComplianceResult,
        intent: BuyerIntentResult,
        engagement_score: float,
    ) -> AttendancePredictionResult:
        strong = rule_signals["strong_commit_hits"]
        weak   = rule_signals["weak_commit_hits"]
        join   = rule_signals["join_signal_hits"]

        # Base probability from compliance + intent
        base = 0.3
        if compliance.email_collected:   base += 0.10
        if compliance.email_verified:    base += 0.10
        if compliance.attendance_commitment_obtained: base += 0.15
        base += (intent.interest_score / 100) * 0.15
        base += (engagement_score / 100) * 0.10
        base += strong * 0.08
        base -= weak   * 0.05
        base += join   * 0.04

        ap = round(max(0.0, min(1.0, base)), 3)
        cls = min(10, max(1, strong * 3 + join - weak))

        risk_factors = []
        if not compliance.email_collected: risk_factors.append("Email not collected")
        if weak > strong: risk_factors.append("Weak commitment language dominates")
        if engagement_score < 40: risk_factors.append("Low engagement throughout call")

        indicators = []
        if compliance.email_verified: indicators.append("Email was verified")
        if join: indicators.append("Joining instructions were discussed")
        if strong: indicators.append(f"{strong} strong commitment expressions")

        confidence = "HIGH" if ap > 0.7 or ap < 0.25 else "MEDIUM" if ap > 0.45 else "LOW"

        return AttendancePredictionResult(
            attendance_probability=ap,
            commitment_language_score=cls,
            commitment_quotes=rule_signals["commit_quotes"],
            joining_instructions_discussed=bool(join),
            time_confirmed=compliance.attendance_commitment_obtained,
            schedule_conflict_mentioned=False,
            follow_through_indicators=indicators,
            risk_factors=risk_factors,
            prediction_confidence=confidence,
            prediction_reasoning="Rule-based prediction (LLM unavailable).",
        )
