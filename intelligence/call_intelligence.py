"""
intelligence/call_intelligence.py
Extracts structured call intelligence using GPT-4o function calling.

Extracts:
  - Pain points (with severity and quote)
  - Budget signals
  - Timeline
  - Decision makers
  - Objections (with handling quality)
  - Competitors mentioned
  - Next steps agreed
  - Buying signals
  - Risk signals
  - Deal risk level

Uses GPT-4o with structured outputs for reliable JSON extraction.
Falls back to Claude if OpenAI unavailable.
"""
from __future__ import annotations
import json
import logging

from config.models_multilingual import (
    CallIntelligence, PainPoint, BudgetSignal, Objection,
    BuyingSignal, EvidenceQuote, SpeakerRole, Language, DealRisk
)
from config.settings import get_settings
from transcription.post_processor import build_full_transcript

logger   = logging.getLogger(__name__)
settings = get_settings()

EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "pain_points": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "severity": {"type": "string", "enum": ["high", "medium", "low"]},
                    "quote": {"type": "string"},
                    "speaker": {"type": "string"},
                    "timestamp": {"type": "string"}
                },
                "required": ["description", "severity", "quote", "speaker", "timestamp"]
            }
        },
        "budget_signals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "mentioned": {"type": "boolean"},
                    "amount": {"type": "string"},
                    "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
                    "quote": {"type": "string"},
                    "timestamp": {"type": "string"}
                }
            }
        },
        "timeline": {"type": "string"},
        "decision_makers": {"type": "array", "items": {"type": "string"}},
        "objections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["price", "timing", "authority", "need", "trust", "other"]},
                    "text": {"type": "string"},
                    "handled": {"type": "boolean"},
                    "handling_quality": {"type": "integer", "minimum": 1, "maximum": 10},
                    "quote": {"type": "string"},
                    "timestamp": {"type": "string"}
                },
                "required": ["type", "text", "handled", "quote", "timestamp"]
            }
        },
        "competitors": {"type": "array", "items": {"type": "string"}},
        "next_steps": {"type": "array", "items": {"type": "string"}},
        "buying_signals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "signal_type": {"type": "string"},
                    "text": {"type": "string"},
                    "strength": {"type": "string", "enum": ["strong", "moderate", "weak"]},
                    "quote": {"type": "string"},
                    "timestamp": {"type": "string"}
                },
                "required": ["signal_type", "text", "strength", "quote", "timestamp"]
            }
        },
        "risk_signals": {"type": "array", "items": {"type": "string"}},
        "deal_risk": {"type": "string", "enum": ["low", "medium", "high"]}
    },
    "required": ["pain_points", "objections", "competitors", "next_steps",
                 "buying_signals", "risk_signals", "deal_risk"]
}

SYSTEM_PROMPT = """You are an expert B2B sales intelligence analyst.
Analyze the following sales call transcript and extract structured intelligence.

The transcript may be in English, Hindi, Hinglish, Tamil, Telugu, Kannada,
Malayalam, Bengali, or Marathi. Understand the full semantic meaning regardless
of language. Extract insights from ALL languages used.

For timestamps, use the [HH:MM:SS] format shown in the transcript.
For quotes, use the original text from the transcript (do not translate).
For descriptions and analysis, respond in English.

Be specific and evidence-grounded. Only extract what is actually present in
the transcript — do not infer or fabricate."""


class CallIntelligenceExtractor:

    def extract(self, utterances: list, duration_sec: float) -> CallIntelligence:
        """
        Extract structured intelligence from call utterances.
        Uses GPT-4o (primary) or Claude (fallback).
        """
        if not utterances:
            return _empty_intelligence()

        transcript = build_full_transcript(utterances)
        n_speakers = len({u.speaker for u in utterances})

        raw = self._call_llm(transcript)
        if raw is None:
            logger.warning("Intelligence extraction returned None — using empty result")
            return _empty_intelligence()

        return _parse_raw(raw)

    def _call_llm(self, transcript: str) -> dict | None:
        # Try GPT-4o with function calling (most reliable for structured output)
        if settings.openai_api_key:
            result = self._extract_openai(transcript)
            if result:
                return result

        # Fallback: Claude
        anthropic_key = getattr(settings, "anthropic_api_key", "")
        if anthropic_key:
            result = self._extract_claude(transcript)
            if result:
                return result

        logger.warning("No LLM API key available for intelligence extraction")
        return None

    def _extract_openai(self, transcript: str) -> dict | None:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=settings.openai_api_key)

            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": f"TRANSCRIPT:\n{transcript[:15000]}"},
                ],
                tools=[{
                    "type": "function",
                    "function": {
                        "name":        "extract_call_intelligence",
                        "description": "Extract structured intelligence from a sales call",
                        "parameters":  EXTRACTION_SCHEMA,
                    }
                }],
                tool_choice={"type": "function", "function": {"name": "extract_call_intelligence"}},
                temperature=0,
            )

            tool_call = resp.choices[0].message.tool_calls[0]
            return json.loads(tool_call.function.arguments)

        except Exception as exc:
            logger.warning(f"GPT-4o intelligence extraction failed: {exc}")
            return None

    def _extract_claude(self, transcript: str) -> dict | None:
        try:
            import anthropic
            key = getattr(settings, "anthropic_api_key", "")
            client = anthropic.Anthropic(api_key=key)

            prompt = (
                f"{SYSTEM_PROMPT}\n\n"
                f"TRANSCRIPT:\n{transcript[:15000]}\n\n"
                f"Respond with a single valid JSON object matching this schema:\n"
                f"{json.dumps(EXTRACTION_SCHEMA, indent=2)}\n\n"
                f"JSON response only, no preamble:"
            )

            msg = client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            text = msg.content[0].text.strip()
            # Extract JSON from response
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            return json.loads(text)

        except Exception as exc:
            logger.warning(f"Claude intelligence extraction failed: {exc}")
            return None


def _parse_raw(raw: dict) -> CallIntelligence:
    """Convert raw LLM JSON response to CallIntelligence model."""

    def make_quote(q: dict | None) -> EvidenceQuote | None:
        if not q:
            return None
        if isinstance(q, str):
            return EvidenceQuote(text=q, speaker=SpeakerRole.UNKNOWN, timestamp="00:00:00")
        return EvidenceQuote(
            text=q.get("quote", q.get("text", "")),
            speaker=SpeakerRole.UNKNOWN,
            timestamp=q.get("timestamp", "00:00:00"),
        )

    pain_points = []
    for pp in raw.get("pain_points", []):
        quote = EvidenceQuote(
            text=pp.get("quote", ""),
            speaker=SpeakerRole.CUSTOMER,
            timestamp=pp.get("timestamp", "00:00:00"),
        )
        pain_points.append(PainPoint(
            description=pp.get("description", ""),
            severity=pp.get("severity", "medium"),
            quote=quote,
        ))

    budget_signals = []
    for bs in raw.get("budget_signals", []):
        q_text = bs.get("quote", "")
        q = EvidenceQuote(text=q_text, speaker=SpeakerRole.CUSTOMER, timestamp=bs.get("timestamp", "00:00:00")) if q_text else None
        budget_signals.append(BudgetSignal(
            mentioned=bs.get("mentioned", bool(q_text)),
            amount=bs.get("amount"),
            sentiment=bs.get("sentiment", "neutral"),
            quote=q,
        ))

    objections = []
    for obj in raw.get("objections", []):
        quote = EvidenceQuote(
            text=obj.get("quote", obj.get("text", "")),
            speaker=SpeakerRole.CUSTOMER,
            timestamp=obj.get("timestamp", "00:00:00"),
        )
        objections.append(Objection(
            type=obj.get("type", "other"),
            text=obj.get("text", ""),
            handled=obj.get("handled", False),
            handling_quality=obj.get("handling_quality"),
            quote=quote,
        ))

    buying_signals = []
    for bs in raw.get("buying_signals", []):
        quote = EvidenceQuote(
            text=bs.get("quote", ""),
            speaker=SpeakerRole.CUSTOMER,
            timestamp=bs.get("timestamp", "00:00:00"),
        )
        buying_signals.append(BuyingSignal(
            signal_type=bs.get("signal_type", "positive_language"),
            text=bs.get("text", ""),
            strength=bs.get("strength", "moderate"),
            quote=quote,
        ))

    deal_risk_str = raw.get("deal_risk", "medium").lower()
    deal_risk = DealRisk(deal_risk_str) if deal_risk_str in ("low", "medium", "high") else DealRisk.MEDIUM

    return CallIntelligence(
        pain_points=pain_points,
        budget_signals=budget_signals,
        timeline=raw.get("timeline"),
        decision_makers=raw.get("decision_makers", []),
        objections=objections,
        competitors=raw.get("competitors", []),
        next_steps=raw.get("next_steps", []),
        buying_signals=buying_signals,
        risk_signals=raw.get("risk_signals", []),
        deal_risk=deal_risk,
    )


def _empty_intelligence() -> CallIntelligence:
    return CallIntelligence(
        pain_points=[], budget_signals=[], objections=[],
        competitors=[], next_steps=[], buying_signals=[],
        risk_signals=[], deal_risk=DealRisk.MEDIUM,
    )
