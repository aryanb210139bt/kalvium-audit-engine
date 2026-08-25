"""
audit/buyer_intent.py
Layer 2 — Buyer Intent Analysis.

Evaluates the PROSPECT's genuine interest level based on question quality,
career seriousness, and intent signals extracted from the transcript.
"""
from __future__ import annotations
import json
import logging

from openai import OpenAI

from config.models import Utterance, Speaker
from config.kalvium_models import BuyerIntentResult, CareerSeriousness
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def _prospect_lines(utterances: list[Utterance]) -> str:
    lines = [
        f"[{int(u.start_time//60):02d}:{int(u.start_time%60):02d}] {u.english_text}"
        for u in utterances
        if u.speaker in (Speaker.STUDENT, Speaker.UNKNOWN)
    ]
    return "\n".join(lines[:80]) if lines else "(no prospect speech detected)"


def _build_intent_prompt(full_transcript: str, prospect_only: str) -> str:
    return f"""You are a behavioral intelligence analyst specializing in ed-tech sales psychology.

CONTEXT:
A Kalvium associate spoke with a potential student (Class 12 PCM/PCMB or dropper)
about registering for a free webinar on Kalvium's B.Tech program.

Genuine interested students:
- Ask about curriculum, placements, fees, or Kalvium's approach
- Mention their stream, marks, or future career goals
- React to AI/job market information with curiosity or concern
- Discuss with parents or mention parental opinion
- Ask how/when to join the webinar
- Show some hesitation or confusion (natural for real decision-making)

Low-intent / fake participants:
- Give vague or non-committal answers
- Never ask about the program at all
- Give one-word confirmations ("yes", "ok", "haan") to everything
- Show no reaction to AI or placement information
- Agree immediately without any questions

PROSPECT UTTERANCES ONLY:
{prospect_only}

FULL TRANSCRIPT (for context):
{full_transcript[:8000]}

TASK:
Evaluate the PROSPECT's genuine interest level. Focus on what the PROSPECT says and
how they respond. Return ONLY valid JSON:

{{
  "interest_score": <0-100>,
  "buying_intent_score": <0-100>,
  "question_depth_score": <0-100>,
  "career_seriousness": "HIGH/MEDIUM/LOW",
  "parent_involved": <true/false>,
  "fee_discussion": <true/false>,
  "placement_interest": <true/false>,
  "questions_asked_by_prospect": ["list all genuine questions prospect asked"],
  "objections_raised": ["list any objections or concerns raised"],
  "confusion_expressed": <true/false>,
  "key_intent_signals": ["behavioral signals indicating intent level"],
  "intent_observations": ["specific behavioral observations"]
}}"""


def _rule_intent_estimate(utterances: list[Utterance]) -> dict:
    """Fallback rule-based intent scoring when LLM unavailable."""
    prospect_utts = [
        u for u in utterances
        if u.speaker in (Speaker.STUDENT, Speaker.UNKNOWN)
    ]
    if not prospect_utts:
        return {
            "interest_score": 10, "buying_intent_score": 10, "question_depth_score": 0,
            "career_seriousness": "LOW", "parent_involved": False,
            "fee_discussion": False, "placement_interest": False,
            "questions_asked_by_prospect": [], "objections_raised": [],
            "confusion_expressed": False,
            "key_intent_signals": ["No prospect speech detected"],
            "intent_observations": ["Zero prospect utterances — high fraud risk"],
        }

    texts = " ".join(u.english_text.lower() + " " + u.native_text.lower()
                     for u in prospect_utts)

    import re
    question_words = len(re.findall(r'\b(what|why|how|when|where|which|kya|kaise|kab|kyun)\b',
                                     texts, re.IGNORECASE))
    parent_words   = len(re.findall(r'\b(parent|father|mother|papa|mummy|mom|dad|ghar)\b',
                                     texts, re.IGNORECASE))
    fee_words      = len(re.findall(r'\b(fee|fees|cost|price|paisa|rupee|lakh|afford)\b',
                                     texts, re.IGNORECASE))
    place_words    = len(re.findall(r'\b(placement|job|salary|company|hire|package)\b',
                                     texts, re.IGNORECASE))

    avg_len = sum(len(u.english_text.split()) for u in prospect_utts) / len(prospect_utts)

    interest  = min(100, question_words * 15 + parent_words * 10 + int(avg_len > 5) * 20)
    intent    = min(100, fee_words * 20 + place_words * 15 + question_words * 10)
    q_depth   = min(100, question_words * 20)

    return {
        "interest_score": interest, "buying_intent_score": intent,
        "question_depth_score": q_depth,
        "career_seriousness": "HIGH" if interest > 60 else "MEDIUM" if interest > 30 else "LOW",
        "parent_involved": parent_words > 0,
        "fee_discussion": fee_words > 0,
        "placement_interest": place_words > 0,
        "questions_asked_by_prospect": [],
        "objections_raised": [],
        "confusion_expressed": False,
        "key_intent_signals": [f"Question words detected: {question_words}"],
        "intent_observations": [f"Avg prospect response length: {avg_len:.1f} words"],
    }


class BuyerIntentAnalyzer:
    """Layer 2: Evaluate genuine interest of the prospect using LLM + rules."""

    def __init__(self):
        if settings.openai_api_key:
            self._client = OpenAI(api_key=settings.openai_api_key)
        else:
            self._client = None

    def analyze(
        self,
        utterances: list[Utterance],
        full_transcript: str,
    ) -> BuyerIntentResult:
        prospect_only = _prospect_lines(utterances)

        if not self._client:
            data = _rule_intent_estimate(utterances)
            return self._parse(data)

        prompt = _build_intent_prompt(full_transcript[:10000], prospect_only)
        try:
            resp = self._client.chat.completions.create(
                model=settings.openai_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=800,
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content or "{}")
            return self._parse(data)
        except Exception as exc:
            logger.error(f"BuyerIntent LLM call failed: {exc}")
            return self._parse(_rule_intent_estimate(utterances))

    def _parse(self, data: dict) -> BuyerIntentResult:
        seriousness_raw = str(data.get("career_seriousness", "LOW")).upper()
        try:
            seriousness = CareerSeriousness(seriousness_raw)
        except ValueError:
            seriousness = CareerSeriousness.LOW

        return BuyerIntentResult(
            interest_score=float(data.get("interest_score", 10)),
            buying_intent_score=float(data.get("buying_intent_score", 10)),
            question_depth_score=float(data.get("question_depth_score", 0)),
            career_seriousness=seriousness,
            parent_involved=bool(data.get("parent_involved", False)),
            fee_discussion=bool(data.get("fee_discussion", False)),
            placement_interest=bool(data.get("placement_interest", False)),
            questions_asked_by_prospect=data.get("questions_asked_by_prospect", []),
            objections_raised=data.get("objections_raised", []),
            confusion_expressed=bool(data.get("confusion_expressed", False)),
            key_intent_signals=data.get("key_intent_signals", []),
            intent_observations=data.get("intent_observations", []),
        )
