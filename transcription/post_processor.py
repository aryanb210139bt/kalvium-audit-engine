"""
transcription/post_processor.py
Transcript post-processing: punctuation, PII redaction, speaker merge,
hallucination filtering, and language tagging.

All steps are local (no API calls) except optional punctuation via GPT-4o-mini.
"""
from __future__ import annotations
import logging
import re
import uuid
from pathlib import Path

from config.models_multilingual import (
    Language, SpeakerRole, Utterance, WordToken,
    SpeakerSegment,
)
from language.detector import detect_language_from_text, detect_hinglish_from_text

logger = logging.getLogger(__name__)

# ── PII patterns for India ─────────────────────────────────────────────────────
PII_PATTERNS: dict[str, re.Pattern] = {
    "aadhaar":  re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),
    "pan":      re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    "phone":    re.compile(r"\b(?:\+91[-\s]?)?[6-9]\d{9}\b"),
    "email":    re.compile(r"\b[\w.+-]+@[\w-]+\.\w{2,}\b"),
    "gst":      re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z][A-Z\d][Z][A-Z\d]\b"),
    "account":  re.compile(r"\b\d{9,18}\b"),
    "ifsc":     re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b"),
}

REDACTION_LABEL: dict[str, str] = {
    "aadhaar":  "[AADHAAR]",
    "pan":      "[PAN]",
    "phone":    "[PHONE]",
    "email":    "[EMAIL]",
    "gst":      "[GST]",
    "account":  "[ACCOUNT]",
    "ifsc":     "[IFSC]",
}


class TranscriptPostProcessor:
    """
    Applies all post-processing steps to raw ASR utterances.
    """

    def __init__(self, use_gpt_punctuation: bool = False):
        self._use_gpt_punct = use_gpt_punctuation

    def process(
        self,
        utterances: list[Utterance],
        speaker_segments: list[SpeakerSegment],
    ) -> list[Utterance]:
        """
        Full post-processing pipeline:
          1. Speaker label correction using diarization segments
          2. Language tagging + Hinglish detection
          3. PII redaction
          4. Utterance deduplication
          5. Question detection
        """
        if not utterances:
            return []

        utterances = self._merge_speaker_labels(utterances, speaker_segments)
        utterances = self._tag_language(utterances)
        utterances = self._redact_pii(utterances)
        utterances = self._deduplicate(utterances)
        utterances = self._tag_questions(utterances)

        logger.info(f"Post-processing complete: {len(utterances)} utterances")
        return utterances

    # ── 1. Speaker label correction ────────────────────────────────────────────

    def _merge_speaker_labels(
        self,
        utterances: list[Utterance],
        diar_segments: list[SpeakerSegment],
    ) -> list[Utterance]:
        """
        Overwrite ASR speaker labels with diarization speaker assignments.
        Uses midpoint of utterance to find the dominant speaker.
        """
        if not diar_segments:
            return utterances

        updated = []
        for utt in utterances:
            mid = (utt.start_time + utt.end_time) / 2
            speaker = _find_speaker_at(diar_segments, mid)
            updated.append(utt.model_copy(update={"speaker": speaker}))
        return updated

    # ── 2. Language tagging ────────────────────────────────────────────────────

    def _tag_language(self, utterances: list[Utterance]) -> list[Utterance]:
        """
        Re-detect language at utterance level using script analysis.
        Overrides model-detected language when script gives higher confidence.
        """
        updated = []
        for utt in utterances:
            if utt.language == Language.UNKNOWN:
                lang = detect_language_from_text(utt.text)
                utt = utt.model_copy(update={"language": lang})
            elif detect_hinglish_from_text(utt.text):
                utt = utt.model_copy(update={"language": Language.HINGLISH})
            updated.append(utt)
        return updated

    # ── 3. PII Redaction ───────────────────────────────────────────────────────

    def _redact_pii(self, utterances: list[Utterance]) -> list[Utterance]:
        updated = []
        for utt in utterances:
            text = utt.text
            eng  = utt.english_text
            for pii_type, pattern in PII_PATTERNS.items():
                label = REDACTION_LABEL[pii_type]
                text = pattern.sub(label, text)
                eng  = pattern.sub(label, eng)
            updated.append(utt.model_copy(update={"text": text, "english_text": eng}))
        return updated

    # ── 4. Deduplication ───────────────────────────────────────────────────────

    def _deduplicate(self, utterances: list[Utterance]) -> list[Utterance]:
        """
        Remove utterances with >70% bigram overlap with the previous utterance
        (caused by ASR overlap from chunking).
        """
        if len(utterances) <= 1:
            return utterances

        result = [utterances[0]]
        for utt in utterances[1:]:
            prev = result[-1]
            # Only deduplicate if same speaker and overlapping time window
            if utt.speaker == prev.speaker and utt.start_time < prev.end_time:
                overlap = _bigram_overlap(prev.text, utt.text)
                if overlap > 0.7:
                    # Keep the longer one
                    if len(utt.text) > len(prev.text):
                        result[-1] = utt
                    continue
            result.append(utt)
        return result

    # ── 5. Question detection ──────────────────────────────────────────────────

    def _tag_questions(self, utterances: list[Utterance]) -> list[Utterance]:
        QUESTION_SIGNALS = [
            "?", "kya", "kyun", "kaise", "kaun", "kab", "kahan",
            "what", "why", "how", "when", "where", "which", "who",
        ]
        updated = []
        for utt in utterances:
            t = utt.text.lower()
            is_q = any(s in t for s in QUESTION_SIGNALS)
            updated.append(utt.model_copy(update={"is_question": is_q}))
        return updated


# ── Helpers ────────────────────────────────────────────────────────────────────

def _find_speaker_at(
    segments: list[SpeakerSegment], t: float
) -> SpeakerRole:
    for seg in segments:
        if seg.start <= t <= seg.end:
            return seg.speaker
    # Find closest segment
    if not segments:
        return SpeakerRole.UNKNOWN
    closest = min(segments, key=lambda s: min(abs(s.start - t), abs(s.end - t)))
    return closest.speaker


def _bigram_overlap(a: str, b: str) -> float:
    """Jaccard overlap of word bigrams between two strings."""
    def bigrams(text: str) -> set:
        words = text.lower().split()
        return {f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)}

    bg_a = bigrams(a)
    bg_b = bigrams(b)
    if not bg_a or not bg_b:
        return 0.0
    intersection = len(bg_a & bg_b)
    union = len(bg_a | bg_b)
    return intersection / union if union > 0 else 0.0


def build_full_transcript(utterances: list[Utterance]) -> str:
    """Format utterances as a readable transcript string for LLM input."""
    lines = []
    for utt in utterances:
        ts = _format_ts(utt.start_time)
        lang_tag = f"[{utt.language.value}]" if utt.language != Language.ENGLISH else ""
        text = utt.english_text or utt.text
        lines.append(f"[{ts}] {utt.speaker.value}: {text} {lang_tag}".rstrip())
    return "\n".join(lines)


def _format_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
