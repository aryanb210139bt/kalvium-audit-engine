"""
intelligence/semantic_chunker.py
Semantic chunking and hierarchical summarization for long calls.

Strategy:
  - Split transcript into topic segments via cosine similarity breakpoints
  - Build 2000-token sliding windows with 200-token overlap for LLM
  - Map-Reduce hierarchical summarization: chunk → section → call summary
  - Optional: index utterances in Qdrant for semantic search

Long calls (>60 min) use Map-Reduce to stay within LLM context limits.
Short calls (<20 min) can be fed to Gemini 2.5 Flash's 1M context directly.
"""
from __future__ import annotations
import logging
import re
from typing import Optional

from config.models_multilingual import Utterance, TopicSegment
from config.settings import get_settings
from transcription.post_processor import build_full_transcript

logger   = logging.getLogger(__name__)
settings = get_settings()

SEGMENT_SIMILARITY_THRESHOLD = 0.70   # below this → new topic
MAX_TOKENS_PER_CHUNK          = 2000
TOKEN_OVERLAP                 = 200
WORDS_PER_TOKEN               = 0.75   # rough English approximation

# Topic titles for common sales call phases
TOPIC_LABELS = [
    "Introduction & Agenda",
    "Discovery & Needs Assessment",
    "Product Demonstration",
    "Pricing & Budget Discussion",
    "Objection Handling",
    "Competitive Comparison",
    "Decision Maker Discussion",
    "Next Steps & Closing",
    "Miscellaneous",
]


class SemanticChunker:
    """
    Segments a call transcript into semantic topic chunks
    and generates hierarchical summaries.
    """

    def __init__(self):
        self._embedder = None
        self._try_load_embedder()

    def _try_load_embedder(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            # multilingual-e5-large handles all Indian languages
            self._embedder = SentenceTransformer("intfloat/multilingual-e5-large")
            logger.info("multilingual-e5-large embedder loaded")
        except Exception as exc:
            logger.info(f"SentenceTransformer not available ({exc}); using rule-based segmentation")

    # ── Public API ─────────────────────────────────────────────────────────────

    def segment(self, utterances: list[Utterance]) -> list[TopicSegment]:
        """
        Split utterances into topic segments.
        Returns list of TopicSegment with utterance_ids assigned.
        """
        if not utterances:
            return []

        if self._embedder is not None:
            return self._segment_by_embedding(utterances)
        return self._segment_by_rules(utterances)

    def build_llm_windows(
        self, utterances: list[Utterance], max_tokens: int = MAX_TOKENS_PER_CHUNK
    ) -> list[str]:
        """
        Build sliding window text blocks for LLM processing.
        Each window is a formatted transcript excerpt fitting within max_tokens.
        """
        full = build_full_transcript(utterances)
        words = full.split()
        max_words = int(max_tokens / WORDS_PER_TOKEN)
        overlap_words = int(TOKEN_OVERLAP / WORDS_PER_TOKEN)
        step = max_words - overlap_words

        windows = []
        i = 0
        while i < len(words):
            chunk = words[i: i + max_words]
            windows.append(" ".join(chunk))
            if i + max_words >= len(words):
                break
            i += step
        return windows

    def summarize_call(
        self,
        utterances: list[Utterance],
        segments:   list[TopicSegment],
        duration_sec: float,
    ) -> str:
        """
        Generate a hierarchical call summary using Map-Reduce.
        Short calls (<25 min) → single pass.
        Long calls → per-segment summaries then synthesize.
        """
        if not utterances:
            return ""

        full_transcript = build_full_transcript(utterances)
        word_count = len(full_transcript.split())

        # Short call: single LLM pass
        if word_count < 4000 or duration_sec < 1500:
            return self._summarize_single(full_transcript, duration_sec)

        # Long call: Map-Reduce
        return self._summarize_mapreduce(utterances, segments, duration_sec)

    # ── Embedding-based segmentation ───────────────────────────────────────────

    def _segment_by_embedding(self, utterances: list[Utterance]) -> list[TopicSegment]:
        import numpy as np

        texts = [u.english_text or u.text for u in utterances]
        # Embed in batches of 32
        embeddings = self._embedder.encode(
            texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True
        )

        # Find semantic breakpoints
        breakpoints = [0]
        for i in range(1, len(embeddings)):
            sim = float(np.dot(embeddings[i - 1], embeddings[i]))
            if sim < SEGMENT_SIMILARITY_THRESHOLD:
                breakpoints.append(i)
        breakpoints.append(len(utterances))

        segments: list[TopicSegment] = []
        for seg_idx in range(len(breakpoints) - 1):
            start_i = breakpoints[seg_idx]
            end_i   = breakpoints[seg_idx + 1]
            utts_in_seg = utterances[start_i:end_i]
            if not utts_in_seg:
                continue

            label = TOPIC_LABELS[seg_idx % len(TOPIC_LABELS)]
            segments.append(TopicSegment(
                segment_id=seg_idx,
                title=label,
                start_time=utts_in_seg[0].start_time,
                end_time=utts_in_seg[-1].end_time,
                utterance_ids=[u.utterance_id for u in utts_in_seg],
            ))

        logger.info(f"Embedding segmentation: {len(segments)} topic segments")
        return segments

    # ── Rule-based segmentation fallback ──────────────────────────────────────

    def _segment_by_rules(self, utterances: list[Utterance]) -> list[TopicSegment]:
        """
        Segment by time: split call into equal-ish blocks of ~5 minutes.
        Labels assigned by position in call (intro → discovery → demo → close).
        """
        if not utterances:
            return []

        total_dur = utterances[-1].end_time - utterances[0].start_time
        block_sec = 300   # 5 minute blocks

        segments: list[TopicSegment] = []
        seg_idx = 0
        current_start = 0.0
        current_utts: list[Utterance] = []

        for utt in utterances:
            if utt.start_time - current_start > block_sec and current_utts:
                segments.append(_make_segment(seg_idx, current_utts))
                seg_idx += 1
                current_utts = []
                current_start = utt.start_time
            current_utts.append(utt)

        if current_utts:
            segments.append(_make_segment(seg_idx, current_utts))

        logger.info(f"Rule-based segmentation: {len(segments)} segments")
        return segments

    # ── Summarization ──────────────────────────────────────────────────────────

    def _summarize_single(self, transcript: str, duration_sec: float) -> str:
        """Single LLM call for short transcripts."""
        return _call_llm_summarize(transcript, duration_sec)

    def _summarize_mapreduce(
        self,
        utterances: list[Utterance],
        segments:   list[TopicSegment],
        duration_sec: float,
    ) -> str:
        """
        Map: summarize each segment independently.
        Reduce: synthesize segment summaries into one call summary.
        """
        utt_map = {u.utterance_id: u for u in utterances}
        seg_summaries = []

        for seg in segments:
            seg_utts = [utt_map[uid] for uid in seg.utterance_ids if uid in utt_map]
            seg_transcript = build_full_transcript(seg_utts)
            if not seg_transcript.strip():
                continue
            summary = _call_llm_summarize(
                seg_transcript, seg.end_time - seg.start_time,
                context=f"This is segment '{seg.title}' of a longer sales call."
            )
            seg_summaries.append(f"[{seg.title}]\n{summary}")
            seg.summary = summary

        if not seg_summaries:
            return ""

        combined = "\n\n".join(seg_summaries)
        return _call_llm_reduce(combined, duration_sec)


# ── LLM calls ──────────────────────────────────────────────────────────────────

def _call_llm_summarize(
    transcript: str,
    duration_sec: float,
    context: str = "",
) -> str:
    """Summarize a transcript using Gemini 2.5 Flash (cheap, fast) or GPT-4o-mini."""
    prompt = (
        f"{context}\n\n"
        f"Summarize this sales call transcript in 3–5 sentences. "
        f"Focus on: what was discussed, key pain points raised, "
        f"objections handled, and outcomes or next steps agreed.\n\n"
        f"TRANSCRIPT:\n{transcript}"
    ).strip()

    # Try Gemini Flash first (cheapest)
    gemini_key = getattr(settings, "gemini_api_key", "")
    if gemini_key:
        try:
            return _gemini_summarize(prompt, gemini_key)
        except Exception as exc:
            logger.debug(f"Gemini summarize failed: {exc}")

    # Fallback: OpenAI
    if settings.openai_api_key:
        try:
            return _openai_summarize(prompt)
        except Exception as exc:
            logger.debug(f"OpenAI summarize failed: {exc}")

    return "[Summary unavailable — no LLM API key configured]"


def _call_llm_reduce(combined_summaries: str, duration_sec: float) -> str:
    """Synthesize multiple segment summaries into a single call summary."""
    prompt = (
        f"You have been given summaries of each segment of a {int(duration_sec/60)}-minute "
        f"sales call. Write a single coherent 5–7 sentence summary of the entire call, "
        f"capturing the full arc from opening to close.\n\n"
        f"SEGMENT SUMMARIES:\n{combined_summaries}"
    )

    gemini_key = getattr(settings, "gemini_api_key", "")
    if gemini_key:
        try:
            return _gemini_summarize(prompt, gemini_key)
        except Exception:
            pass

    if settings.openai_api_key:
        try:
            return _openai_summarize(prompt)
        except Exception:
            pass

    return combined_summaries[:1000]


def _gemini_summarize(prompt: str, api_key: str) -> str:
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text.strip()


def _openai_summarize(prompt: str) -> str:
    from openai import OpenAI
    from config.settings import get_settings
    s = get_settings()
    client = OpenAI(api_key=s.openai_api_key)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


def _make_segment(seg_idx: int, utts: list[Utterance]) -> TopicSegment:
    label = TOPIC_LABELS[seg_idx % len(TOPIC_LABELS)]
    return TopicSegment(
        segment_id=seg_idx,
        title=label,
        start_time=utts[0].start_time,
        end_time=utts[-1].end_time,
        utterance_ids=[u.utterance_id for u in utts],
    )
