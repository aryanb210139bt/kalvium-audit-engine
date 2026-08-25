"""
intelligence/event_detector.py
Step 6a — Converts raw utterances → structured conversation events.
This is the critical pre-processing step before the audit engine:
instead of feeding raw transcript to LLM, we feed structured events.

Approach: keyword + pattern matching with weighted confidence scoring.
"""
from __future__ import annotations
import re
import logging
from datetime import timedelta

from config.models import (
    Utterance, StructuredEvent, ConversationEvent, Speaker, Sentiment
)

logger = logging.getLogger(__name__)

# ── Keyword pattern maps ───────────────────────────────────────────────────────
# Each event has a list of regex patterns. Match count → confidence score.

EVENT_PATTERNS: dict[ConversationEvent, list[str]] = {
    ConversationEvent.COUNSELLOR_INTRO: [
        r"\bmy name is\b", r"\bi am\b.*\bcounsell", r"\bwelcome\b",
        r"\bkalvium\b.*\btoday\b", r"\bintroduce myself\b",
        r"\bhi\b.*\bpleasure\b", r"\bglad.*you.*joined\b",
    ],
    ConversationEvent.KALVIUM_EXPLANATION: [
        r"\bkalvium\b", r"\bwork.integrated\b", r"\bwork integrated\b",
        r"\bindustry.ready\b", r"\bpractical.*curriculum\b",
        r"\breal.*world.*experience\b", r"\bstartup.*culture\b",
    ],
    ConversationEvent.TRADITIONAL_COLLEGE_COMPARISON: [
        r"\btraditional college\b", r"\bconventional.*college\b",
        r"\bcompared to\b.*\bcollege\b", r"\buniversity\b.*\bdifferent\b",
        r"\bbetter than\b.*\bcollege\b", r"\bdifference\b.*\bcollege\b",
        r"\bengineering college\b", r"\bnit\b|\biit\b|\bvit\b",
    ],
    ConversationEvent.CAREER_OUTCOMES: [
        r"\bcareer\b", r"\bjob\b.*\bafter\b", r"\bpackage\b",
        r"\blakh\b", r"\bsalary\b", r"\bplacement\b.*\brecord\b",
        r"\bgrowth\b.*\bopportunity\b", r"\bindustry.*exposure\b",
    ],
    ConversationEvent.REPO_SHOWN: [
        r"\bgithub\b", r"\brepository\b|\brepo\b", r"\bcode\b.*\bproject\b",
        r"\bportfolio\b", r"\bshowing.*project\b", r"\blet me show.*code\b",
        r"\bcommit\b.*\bgithub\b",
    ],
    ConversationEvent.PPT_PRESENTED: [
        r"\bpresentation\b|\bslide\b|\bppt\b",
        r"\blet me show\b", r"\bsharing.*screen\b|\bscreen.*share\b",
        r"\bnext slide\b", r"\bthis slide\b",
    ],
    ConversationEvent.PLACEMENTS_DISCUSSION: [
        r"\bplacement\b", r"\bcompan(y|ies)\b.*\bhire\b|\bhire\b.*\bcompan\b",
        r"\brecruiter\b", r"\bcampus.*recruit\b",
        r"\bjob offer\b", r"\bselected by\b", r"\bstudent.*placed\b",
    ],
    ConversationEvent.FEE_STRUCTURE: [
        r"\bfee\b", r"\bcost\b", r"\bprice\b|\bpricing\b",
        r"\bhow much\b", r"\bpayment\b", r"\bemi\b", r"\binstalment\b",
        r"\baffordabl\b", r"\bexpensive\b", r"\blakh\b.*\bfee\b",
    ],
    ConversationEvent.ROI_EXPLANATION: [
        r"\broi\b|\breturn on investment\b",
        r"\bworth.*investment\b|\binvestment.*worth\b",
        r"\bbreak.*even\b", r"\bpayback\b",
        r"\blong.term.*value\b", r"\bcompare.*fee.*salary\b",
    ],
    ConversationEvent.DISCOVERY_QUESTIONS: [
        r"\bwhat.*interest\b|\bwhat.*passion\b",
        r"\bwhat.*goal\b|\bwhat.*plan\b",
        r"\bwhere.*see\b.*\b5 year\b|\bfuture.*plan\b",
        r"\btell me about\b", r"\bwhat.*expect\b",
        r"\bwhy.*kalvium\b", r"\bhow.*hear.*us\b",
        r"\bwhat.*background\b", r"\bcurrent.*study\b",
    ],
    ConversationEvent.CLOSING_ATTEMPT: [
        r"\bnext step\b", r"\bhow.*apply\b|\bapplication.*process\b",
        r"\bready.*enroll\b|\bready.*join\b",
        r"\bdeadline\b|\blast.*date\b",
        r"\blimited.*seat\b|\bseat.*fill\b",
        r"\bstart.*date\b|\bbatch.*start\b",
    ],
    ConversationEvent.OBJECTION_RAISED: [
        r"\btoo expensive\b|\btoo costly\b|\btoo high\b",
        r"\bnot sure\b|\bunsure\b", r"\bworried\b|\bconcerned\b",
        r"\bwhat if\b.*\bjob\b", r"\bnot aicte\b|\brecognised\b",
        r"\bnew college\b", r"\brisky\b|\brisk\b",
        r"\btraditional.*safer\b|\bsafer.*traditional\b",
        r"\bparent.*not agree\b|\bmom.*not agree\b|\bdad.*not agree\b",
    ],
    ConversationEvent.OBJECTION_HANDLED: [
        r"\bi understand\b.*\bconcern\b", r"\bgreat question\b",
        r"\blet me explain\b", r"\bactually\b.*\bwhat happens\b",
        r"\bour student\b.*\bplaced\b", r"\bno need to worry\b",
        r"\bfully approved\b|\baicte.*approved\b",
        r"\bscholarship\b.*\bavailable\b",
    ],
}

# Sentiment keyword maps (simple rule-based, XLM-RoBERTa used in full pipeline)
POSITIVE_WORDS = {
    "great", "excellent", "amazing", "love", "excited", "interested",
    "sounds good", "perfect", "wonderful", "fantastic", "definitely",
    "absolutely", "yes", "sure", "happy", "glad", "thank",
}
NEGATIVE_WORDS = {
    "expensive", "costly", "worried", "concern", "doubt", "unsure",
    "not sure", "risky", "risk", "problem", "issue", "difficult",
    "can't afford", "too much", "no", "don't", "won't", "difficult",
}


def seconds_to_timestamp(seconds: float) -> str:
    td = timedelta(seconds=int(seconds))
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def detect_sentiment(text: str) -> Sentiment:
    """Rule-based sentiment detection. XLM-RoBERTa used in full ML layer."""
    text_lower = text.lower()
    pos = sum(1 for w in POSITIVE_WORDS if w in text_lower)
    neg = sum(1 for w in NEGATIVE_WORDS if w in text_lower)
    if pos > neg:
        return Sentiment.POSITIVE
    elif neg > pos:
        return Sentiment.NEGATIVE
    return Sentiment.NEUTRAL


def detect_events(text: str) -> list[tuple[ConversationEvent, float]]:
    """
    Detect which conversation events are present in a text segment.
    Returns list of (event, confidence) sorted by confidence desc.
    """
    text_lower = text.lower()
    detected: list[tuple[ConversationEvent, float]] = []

    for event, patterns in EVENT_PATTERNS.items():
        matches = sum(1 for p in patterns if re.search(p, text_lower))
        if matches > 0:
            confidence = min(1.0, matches / max(len(patterns) * 0.3, 1))
            detected.append((event, round(confidence, 2)))

    return sorted(detected, key=lambda x: x[1], reverse=True)


class ConversationEventDetector:
    """
    Converts utterances into structured conversation events.
    This is the key pre-processing step before feeding into the audit engine.
    """

    def detect(self, utterances: list[Utterance]) -> list[StructuredEvent]:
        """
        Process all utterances and produce structured events.
        Each utterance may map to zero, one, or multiple events.
        """
        events: list[StructuredEvent] = []

        for utterance in utterances:
            text = utterance.english_text or utterance.native_text
            if not text.strip():
                continue

            detected = detect_events(text)
            sentiment = detect_sentiment(text)

            if not detected:
                # Still create a GENERIC event for talk ratio analysis
                continue

            # Top event per utterance (can extend to multi-label if needed)
            top_event, confidence = detected[0]

            structured = StructuredEvent(
                utterance_id=utterance.utterance_id,
                speaker=utterance.speaker,
                event=top_event,
                timestamp=seconds_to_timestamp(utterance.start_time),
                sentiment=sentiment,
                native_text=utterance.native_text,
                english_text=utterance.english_text,
                confidence=confidence,
            )
            events.append(structured)

            if len(detected) > 1:
                # Log secondary events for debugging
                secondary = [e.value for e, _ in detected[1:3]]
                logger.debug(f"  Secondary events for {utterance.utterance_id}: {secondary}")

        logger.info(f"Detected {len(events)} structured events from {len(utterances)} utterances")
        return events
