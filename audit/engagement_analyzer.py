"""
audit/engagement_analyzer.py
Layer 3 — Engagement Analysis.

Measures conversation quality: talk ratio, passive behavior, emotional variation,
energy level, and unnatural agreement patterns.
"""
from __future__ import annotations
import json
import logging
import re

from openai import OpenAI

from config.models import Utterance, TalkRatioResult, Speaker
from config.kalvium_models import EngagementResult, EngagementLevel, ConversationEnergy
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# One-word / filler responses that indicate passive engagement
_PASSIVE_PATTERNS = [
    r"^\s*(yes|no|ok|okay|haan|nahi|theek|theek hai|hmm|uh|uhh|achha|acha|right|sure|fine|got it)\s*[.?!]?\s*$",
    r"^\s*(ha|haa|bilkul|zaroor|sahi|agree|understood)\s*[.?!]?\s*$",
]
_PASSIVE_RE = re.compile("|".join(_PASSIVE_PATTERNS), re.IGNORECASE)


def _count_passive(utterances: list[Utterance]) -> tuple[int, int, int]:
    """Returns (passive_count, one_word_count, multi_sentence_count) for prospect."""
    passive = one_word = multi_sent = 0
    for u in utterances:
        if u.speaker not in (Speaker.STUDENT, Speaker.UNKNOWN):
            continue
        text = u.english_text.strip()
        words = text.split()
        if _PASSIVE_RE.match(text):
            passive += 1
        if len(words) <= 2:
            one_word += 1
        elif len(words) >= 15:
            multi_sent += 1
    return passive, one_word, multi_sent


def _build_engagement_prompt(
    full_transcript: str,
    talk_ratio: TalkRatioResult | None,
) -> str:
    tr_info = ""
    if talk_ratio:
        tr_info = (
            f"TALK RATIO — Associate: {talk_ratio.counsellor_pct:.0f}% | "
            f"Prospect: {talk_ratio.student_pct:.0f}% | "
            f"Parent: {talk_ratio.parent_pct:.0f}%"
        )

    return f"""You are a conversation intelligence analyst measuring call quality and genuine engagement.

METRICS TO ANALYZE:
- Talk ratio: Who dominates the conversation?
- Response depth: Are prospect's answers multi-sentence or one-word?
- Passive behavior: "haan", "okay", "hmm", "theek hai", "yes yes" without substance
- Emotional variation: Does tone change? Does prospect show interest/surprise/concern?
- Energy level: Is conversation alive and dynamic, or flat and scripted?
- Interruptions: Natural conversations have natural interruptions
- Unnatural agreement: Prospect agrees instantly without processing or questioning

{tr_info}

FULL TRANSCRIPT:
{full_transcript[:10000]}

TASK:
Analyze the QUALITY and AUTHENTICITY of engagement. A genuine prospect engages, reacts,
and contributes naturally. A fake booking has a passive, robotic prospect who agrees
to everything without real engagement.

Return ONLY valid JSON:
{{
  "engagement_score": <0-100>,
  "attention_level": "HIGH/MEDIUM/LOW/ABSENT",
  "emotional_engagement": "ACTIVE/MODERATE/PASSIVE/ROBOTIC",
  "passive_responses_count": <integer>,
  "one_word_replies_count": <integer>,
  "multi_sentence_responses_count": <integer>,
  "emotional_variation_detected": <true/false>,
  "conversation_energy": "HIGH/MEDIUM/FLAT/SCRIPTED",
  "associate_dominated": <true/false>,
  "unnatural_agreement_patterns": <true/false>,
  "engagement_observations": ["specific behavioral notes"],
  "concerning_patterns": ["patterns suggesting disengagement or fakeness"]
}}"""


class EngagementAnalyzer:
    """Layer 3: Measure conversation engagement quality."""

    def __init__(self):
        if settings.openai_api_key:
            self._client = OpenAI(api_key=settings.openai_api_key)
        else:
            self._client = None

    def analyze(
        self,
        utterances: list[Utterance],
        full_transcript: str,
        talk_ratio: TalkRatioResult | None = None,
    ) -> EngagementResult:
        passive, one_word, multi_sent = _count_passive(utterances)

        if not self._client:
            return self._rule_based(utterances, talk_ratio, passive, one_word, multi_sent)

        prompt = _build_engagement_prompt(full_transcript[:10000], talk_ratio)
        try:
            resp = self._client.chat.completions.create(
                model=settings.openai_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=700,
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content or "{}")
            # Override with deterministic counts
            data["passive_responses_count"] = max(
                int(data.get("passive_responses_count", 0)), passive
            )
            data["one_word_replies_count"] = max(
                int(data.get("one_word_replies_count", 0)), one_word
            )
            data["multi_sentence_responses_count"] = max(
                int(data.get("multi_sentence_responses_count", 0)), multi_sent
            )
            return self._parse(data, talk_ratio)
        except Exception as exc:
            logger.error(f"Engagement LLM call failed: {exc}")
            return self._rule_based(utterances, talk_ratio, passive, one_word, multi_sent)

    # ── parsers ───────────────────────────────────────────────────────────────

    def _parse(self, data: dict, talk_ratio: TalkRatioResult | None) -> EngagementResult:
        dominated = (
            bool(data.get("associate_dominated", False))
            or (talk_ratio is not None and talk_ratio.counsellor_pct > 75)
        )

        engagement_raw = str(data.get("emotional_engagement", "PASSIVE")).upper()
        try:
            emotional_engagement = EngagementLevel(engagement_raw)
        except ValueError:
            emotional_engagement = EngagementLevel.PASSIVE

        energy_raw = str(data.get("conversation_energy", "FLAT")).upper()
        try:
            conversation_energy = ConversationEnergy(energy_raw)
        except ValueError:
            conversation_energy = ConversationEnergy.FLAT

        return EngagementResult(
            engagement_score=float(data.get("engagement_score", 20)),
            attention_level=str(data.get("attention_level", "LOW")).upper(),
            emotional_engagement=emotional_engagement,
            passive_responses_count=int(data.get("passive_responses_count", 0)),
            one_word_replies_count=int(data.get("one_word_replies_count", 0)),
            multi_sentence_responses_count=int(data.get("multi_sentence_responses_count", 0)),
            emotional_variation_detected=bool(data.get("emotional_variation_detected", False)),
            conversation_energy=conversation_energy,
            associate_dominated=dominated,
            unnatural_agreement_patterns=bool(data.get("unnatural_agreement_patterns", False)),
            engagement_observations=data.get("engagement_observations", []),
            concerning_patterns=data.get("concerning_patterns", []),
        )

    def _rule_based(
        self,
        utterances: list[Utterance],
        talk_ratio: TalkRatioResult | None,
        passive: int,
        one_word: int,
        multi_sent: int,
    ) -> EngagementResult:
        prospect_utts = [
            u for u in utterances if u.speaker in (Speaker.STUDENT, Speaker.UNKNOWN)
        ]
        total_prospect = len(prospect_utts) or 1
        passive_ratio  = passive / total_prospect
        one_word_ratio = one_word / total_prospect

        dominated = (talk_ratio is not None and talk_ratio.counsellor_pct > 75)

        # Score based on passive ratio
        if passive_ratio > 0.7:
            score, level, energy = 15, "ABSENT", ConversationEnergy.SCRIPTED
            emotional = EngagementLevel.ROBOTIC
        elif passive_ratio > 0.5:
            score, level, energy = 30, "LOW", ConversationEnergy.FLAT
            emotional = EngagementLevel.PASSIVE
        elif passive_ratio > 0.3:
            score, level, energy = 55, "MEDIUM", ConversationEnergy.MEDIUM
            emotional = EngagementLevel.MODERATE
        else:
            score, level, energy = 75, "HIGH", ConversationEnergy.HIGH
            emotional = EngagementLevel.ACTIVE

        if dominated:
            score = max(0, score - 15)

        return EngagementResult(
            engagement_score=float(score),
            attention_level=level,
            emotional_engagement=emotional,
            passive_responses_count=passive,
            one_word_replies_count=one_word,
            multi_sentence_responses_count=multi_sent,
            emotional_variation_detected=multi_sent > 2,
            conversation_energy=energy,
            associate_dominated=dominated,
            unnatural_agreement_patterns=passive_ratio > 0.6,
            engagement_observations=[
                f"Passive response ratio: {passive_ratio:.0%}",
                f"One-word reply ratio: {one_word_ratio:.0%}",
                f"Multi-sentence responses: {multi_sent}",
            ],
            concerning_patterns=(
                ["High passive response rate — likely fake booking"]
                if passive_ratio > 0.6 else []
            ),
        )
