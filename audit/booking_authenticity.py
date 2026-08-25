"""
audit/booking_authenticity.py
Layer 4 — Booking Authenticity & Fraud Detection.

The most critical layer. Uses:
- Tiered flag system (Tier 1 hard disqualifiers, Tier 2 strong indicators, Tier 3 weak)
- GPT-4o behavioral analysis
- Email verification analysis
- Conversation psychology analysis (confusion→curiosity→evaluation→commitment arc)
"""
from __future__ import annotations
import json
import logging
import re

from openai import OpenAI

from config.models import Utterance, TalkRatioResult, Speaker
from config.kalvium_models import (
    BookingAuthenticityResult, EmailAnalysis, FraudRiskLevel
)
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ── Tier-based flag signals ───────────────────────────────────────────────────

# Tier 1: Any single hit → HIGH RISK immediately
_TIER1_CHECKS = {
    "no_prospect_speech": "Prospect never speaks beyond minimal words",
    "associate_monologue": "Associate talks > 90% of the call",
    "zero_qualification": "No qualification step — stream/class not discussed",
    "no_email_collection": "Email ID never collected during entire call",
    "ultra_short_call": "Call duration < 90 seconds with claimed registration",
}

# Tier 2: 3+ hits → HIGH RISK, 1-2 → MEDIUM
_TIER2_PATTERNS = {
    "overfamiliar_tone": [
        r"\bbhai\b", r"\byaar\b", r"\bdude\b", r"\bpagal\b", r"\bhaha\b",
        r"\blol\b", r"\bkya kar raha hai\b", r"\barrey\b.*\byaar\b",
        r"\bbhai.*sun\b", r"\bsunna yaar\b",
    ],
    "email_not_verified": [
        r"email.*\@",   # has email but no verification following
    ],
    "instant_agreement": [
        r"(yes|haan|ok|okay|sure|bilkul|zaroor|theek).{0,20}(yes|haan|ok|sure|bilkul){2,}",
    ],
    "no_webinar_joining_discussed": [
        r"\bteams.*link\b", r"\bjoin.*link\b", r"\bhow.*join\b", r"\blink.*send\b",
    ],
    "scripted_rhythm": [],   # detected by LLM pattern analysis
}

# Tier 3: 5+ hits → SUSPICIOUS
_TIER3_PATTERNS = {
    "minimal_prospect_speech": [],
    "no_career_context": [],
    "no_followup_questions": [],
    "vague_commitment": [
        r"\b(will try|maybe|will see|dekh lenge|try karenge|dekhte hain)\b",
    ],
    "no_parent_mention_for_student": [],
}


def _check_tier1(
    utterances: list[Utterance],
    talk_ratio: TalkRatioResult | None,
    duration_seconds: float,
    compliance_email_collected: bool,
    compliance_qualified: bool,
) -> list[str]:
    """Return list of triggered Tier 1 flags."""
    flags: list[str] = []

    prospect_utts = [u for u in utterances if u.speaker in (Speaker.STUDENT, Speaker.UNKNOWN)]
    prospect_word_count = sum(len(u.english_text.split()) for u in prospect_utts)

    if prospect_word_count < 20:
        flags.append("no_prospect_speech")
    if talk_ratio and talk_ratio.counsellor_pct > 90:
        flags.append("associate_monologue")
    if not compliance_qualified:
        flags.append("zero_qualification")
    if not compliance_email_collected:
        flags.append("no_email_collection")
    if duration_seconds < 90 and prospect_word_count < 30:
        flags.append("ultra_short_call")

    return flags


def _check_tier2(utterances: list[Utterance], full_text: str) -> list[str]:
    """Return list of triggered Tier 2 flags."""
    flags: list[str] = []

    # Overfamiliar tone
    if any(re.search(p, full_text, re.IGNORECASE)
           for p in _TIER2_PATTERNS["overfamiliar_tone"]):
        flags.append("overfamiliar_tone")

    # Webinar joining never discussed
    if not any(re.search(p, full_text, re.IGNORECASE)
               for p in _TIER2_PATTERNS["no_webinar_joining_discussed"]):
        flags.append("no_webinar_joining_discussed")

    # Instant repeated agreement (3+ consecutive positives)
    consecutive = 0
    max_consecutive = 0
    for u in utterances:
        if u.speaker in (Speaker.STUDENT, Speaker.UNKNOWN):
            text = u.english_text.lower().strip()
            if re.match(r"^(yes|ok|okay|haan|sure|right|bilkul|theek|fine)\s*[.!]?$", text):
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0
    if max_consecutive >= 4:
        flags.append("instant_agreement")

    return flags


def _email_analysis(full_text: str) -> EmailAnalysis:
    """Deterministic email verification check."""
    has_email   = bool(re.search(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}', full_text))
    has_repeat  = bool(re.search(
        r'(repeat|spell|confirm|verify|sahi hai|dobara|correct email|that right)',
        full_text, re.IGNORECASE
    ))
    has_joining = bool(re.search(
        r'(teams.*link|joining.*link|how.*join|link.*send|check.*mail|inbox)',
        full_text, re.IGNORECASE
    ))
    has_confirm = bool(re.search(
        r'(got it|received|check.*email|mail.*aaya|confirmed|yes.*email)',
        full_text, re.IGNORECASE
    ))

    if not has_email:
        risk = FraudRiskLevel.HIGH
    elif not has_repeat:
        risk = FraudRiskLevel.MEDIUM
    else:
        risk = FraudRiskLevel.LOW

    return EmailAnalysis(
        email_requested=has_email,
        email_repeated_by_associate=has_repeat,
        email_confirmed_by_prospect=has_confirm,
        webinar_joining_discussed=has_joining,
        email_risk_level=risk,
    )


def _build_authenticity_prompt(
    full_transcript: str,
    duration_seconds: float,
    talk_ratio: TalkRatioResult | None,
) -> str:
    tr_str = ""
    if talk_ratio:
        tr_str = (
            f"CALL STATS: Duration={duration_seconds:.0f}s | "
            f"Associate={talk_ratio.counsellor_pct:.0f}% | "
            f"Prospect={talk_ratio.student_pct:.0f}%"
        )

    return f"""You are a fraud detection specialist trained to identify fake, forced, and
non-genuine registrations in ed-tech sales calls.

CONTEXT: Kalvium associates register students (Class 12 PCM/PCMB, droppers) for free webinars.
Some associates game their targets by registering friends, family, or coached individuals.

GENUINE BOOKING SIGNS:
- Natural confusion, questions, and genuine evaluation
- Real objections (cost, time, relevance, parent approval needed)
- Associate answers objections genuinely
- Proper email collection AND verbal verification
- Natural conversation rhythm with interruptions
- Prospect mentions personal academic context (board exams, coaching, parents)
- Psychological arc: confusion → curiosity → evaluation → commitment

FAKE/SUSPICIOUS SIGNS:
- Prospect gives only "yes", "ok", "haan", "theek hai" throughout
- No email collected OR email not verified/repeated back
- Call ends very fast with registration supposedly complete
- Associate does 85%+ of all talking
- Zero questions from prospect
- Overfamiliar/casual tone (nicknames, jokes, bhai/yaar language)
- No natural objections whatsoever
- Instant agreement to everything including major life decisions
- No joining instructions discussed
- Scripted/coached-sounding responses
- No personal academic context shared
- Conversation psychology arc is skipped entirely

{tr_str}

TRANSCRIPT:
{full_transcript[:10000]}

TASK: Determine with high precision whether this registration appears genuine or fake/suspicious.
Be specific about behavioral patterns.

Return ONLY valid JSON:
{{
  "booking_authenticity_score": <0-100, 100=definitely genuine>,
  "fake_booking_probability": <0.0-1.0>,
  "fraud_risk_level": "LOW/MEDIUM/HIGH",
  "red_flags": ["specific red flags with transcript evidence"],
  "suspicious_patterns": ["concerning patterns"],
  "relationship_indicators": ["signs of personal relationship between associate and prospect"],
  "call_duration_risk": "LOW/MEDIUM/HIGH — <brief explanation>",
  "buying_journey_present": <true/false>,
  "natural_objections_present": <true/false>,
  "coaching_indicators": ["signs prospect was coached or is known to associate"],
  "authenticity_reasoning": "<detailed paragraph explaining conclusion>"
}}"""


class BookingAuthenticityAnalyzer:
    """
    Layer 4: Fake booking detection using tiered flags + GPT-4o behavioral analysis.
    This is the most critical layer for fraud detection.
    """

    def __init__(self):
        if settings.openai_api_key:
            self._client = OpenAI(api_key=settings.openai_api_key)
        else:
            self._client = None

    def analyze(
        self,
        utterances: list[Utterance],
        full_transcript: str,
        duration_seconds: float,
        talk_ratio: TalkRatioResult | None = None,
        compliance_email_collected: bool = False,
        compliance_qualified: bool = False,
    ) -> BookingAuthenticityResult:
        full_text = full_transcript.lower()

        # Deterministic tier checks
        tier1 = _check_tier1(
            utterances, talk_ratio, duration_seconds,
            compliance_email_collected, compliance_qualified
        )
        tier2 = _check_tier2(utterances, full_text)
        email_anal = _email_analysis(full_transcript)

        # If hard disqualifiers hit — skip LLM, return HIGH RISK immediately
        if len(tier1) >= 2 and not self._client:
            return self._build_from_tiers(tier1, tier2, [], email_anal, duration_seconds, talk_ratio)

        if not self._client:
            return self._build_from_tiers(tier1, tier2, [], email_anal, duration_seconds, talk_ratio)

        # LLM-based behavioral analysis
        prompt = _build_authenticity_prompt(full_transcript, duration_seconds, talk_ratio)
        try:
            resp = self._client.chat.completions.create(
                model=settings.openai_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000,
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content or "{}")
            return self._parse_with_tiers(data, tier1, tier2, email_anal)
        except Exception as exc:
            logger.error(f"BookingAuthenticity LLM call failed: {exc}")
            return self._build_from_tiers(tier1, tier2, [], email_anal, duration_seconds, talk_ratio)

    # ── result builders ───────────────────────────────────────────────────────

    def _parse_with_tiers(
        self,
        data: dict,
        tier1: list[str],
        tier2: list[str],
        email_anal: EmailAnalysis,
    ) -> BookingAuthenticityResult:
        # Tier flags can only make score WORSE, not better
        llm_auth_score = float(data.get("booking_authenticity_score", 50))
        llm_fp         = float(data.get("fake_booking_probability", 0.5))

        # Tier penalties
        tier1_penalty = len(tier1) * 20
        tier2_penalty = len(tier2) * 8
        email_penalty = (
            20 if not email_anal.email_requested else
            10 if not email_anal.email_repeated_by_associate else 0
        )

        auth_score = max(0, llm_auth_score - tier1_penalty - tier2_penalty - email_penalty)
        fp = min(1.0, llm_fp + len(tier1) * 0.15 + len(tier2) * 0.06)

        # Fraud risk
        if len(tier1) >= 1 or fp > 0.65:
            risk = FraudRiskLevel.HIGH
        elif len(tier2) >= 2 or fp > 0.35:
            risk = FraudRiskLevel.MEDIUM
        else:
            risk = FraudRiskLevel.LOW

        red_flags = list(data.get("red_flags", []))
        for t in tier1:
            red_flags.insert(0, f"[TIER-1] {_TIER1_CHECKS.get(t, t)}")
        for t in tier2:
            red_flags.append(f"[TIER-2] {t.replace('_', ' ').title()}")

        raw_risk = str(data.get("fraud_risk_level", "HIGH")).upper()
        try:
            llm_risk = FraudRiskLevel(raw_risk)
        except ValueError:
            llm_risk = FraudRiskLevel.MEDIUM
        # Take the worse of LLM and tier-computed risk
        final_risk = risk if risk == FraudRiskLevel.HIGH else llm_risk

        return BookingAuthenticityResult(
            booking_authenticity_score=round(auth_score, 1),
            fake_booking_probability=round(fp, 3),
            fraud_risk_level=final_risk,
            red_flags=red_flags,
            suspicious_patterns=data.get("suspicious_patterns", []),
            relationship_indicators=data.get("relationship_indicators", []),
            email_analysis=email_anal,
            call_duration_risk=str(data.get("call_duration_risk", "UNKNOWN")),
            buying_journey_present=bool(data.get("buying_journey_present", False)),
            natural_objections_present=bool(data.get("natural_objections_present", False)),
            coaching_indicators=data.get("coaching_indicators", []),
            authenticity_reasoning=str(data.get("authenticity_reasoning", "")),
            tier1_flags=len(tier1),
            tier2_flags=len(tier2),
            tier3_flags=0,
        )

    def _build_from_tiers(
        self,
        tier1: list[str],
        tier2: list[str],
        tier3: list[str],
        email_anal: EmailAnalysis,
        duration_seconds: float,
        talk_ratio: TalkRatioResult | None,
    ) -> BookingAuthenticityResult:
        # Rule-only scoring
        base_auth = 70.0
        base_fp   = 0.2

        base_auth -= len(tier1) * 20
        base_fp   += len(tier1) * 0.18
        base_auth -= len(tier2) * 8
        base_fp   += len(tier2) * 0.06

        if not email_anal.email_requested:
            base_auth -= 20; base_fp += 0.15
        elif not email_anal.email_repeated_by_associate:
            base_auth -= 10; base_fp += 0.08

        auth_score = max(0, min(100, base_auth))
        fp         = max(0, min(1.0, base_fp))

        if len(tier1) >= 1 or fp > 0.65:
            risk = FraudRiskLevel.HIGH
        elif len(tier2) >= 2 or fp > 0.35:
            risk = FraudRiskLevel.MEDIUM
        else:
            risk = FraudRiskLevel.LOW

        red_flags = [
            f"[TIER-1] {_TIER1_CHECKS.get(t, t)}" for t in tier1
        ] + [
            f"[TIER-2] {t.replace('_', ' ').title()}" for t in tier2
        ]

        return BookingAuthenticityResult(
            booking_authenticity_score=round(auth_score, 1),
            fake_booking_probability=round(fp, 3),
            fraud_risk_level=risk,
            red_flags=red_flags,
            suspicious_patterns=[],
            relationship_indicators=[],
            email_analysis=email_anal,
            call_duration_risk=(
                f"HIGH — {duration_seconds:.0f}s is very short" if duration_seconds < 90
                else f"LOW — {duration_seconds:.0f}s call duration"
            ),
            buying_journey_present=False,
            natural_objections_present=False,
            coaching_indicators=[],
            authenticity_reasoning="Rule-based analysis only (LLM unavailable).",
            tier1_flags=len(tier1),
            tier2_flags=len(tier2),
            tier3_flags=len(tier3),
        )
