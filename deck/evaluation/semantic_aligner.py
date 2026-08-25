"""
deck/evaluation/semantic_aligner.py
Semantic slide alignment engine.

For each deck criterion, determines whether the counsellor covered it
using sliding-window cosine similarity — no keywords, pure semantics.
Credit is given for paraphrased explanations, not just exact wording.
"""
from __future__ import annotations
import logging
import math
from dataclasses import dataclass, field
from typing import Optional

from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

WINDOW_TOKENS   = 250    # ~2 minutes of speech
WINDOW_STEP     = 120    # 50% overlap
EMBED_MODEL     = "text-embedding-3-small"


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class CoverageResult:
    criterion_id:     str
    label:            str
    importance:       str
    slide_number:     int
    covered:          bool
    best_similarity:  float
    best_window_idx:  int
    detection_mode:   str   # "direct" | "paraphrase" | "none"
    threshold:        float


@dataclass
class DeckCoverageReport:
    deck_id:          str
    session_id:       str
    coverage_score:   float          # 0-100
    must_covered:     int
    must_total:       int
    should_covered:   int
    should_total:     int
    can_covered:      int
    can_total:        int
    results:          list[CoverageResult] = field(default_factory=list)
    missed_must:      list[str] = field(default_factory=list)
    missed_should:    list[str] = field(default_factory=list)


# ── Cosine similarity (no scipy dependency) ───────────────────────────────────

def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot  = sum(x * y for x, y in zip(a, b))
    na   = math.sqrt(sum(x * x for x in a))
    nb   = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ── Windowing ─────────────────────────────────────────────────────────────────

def _build_windows(utterances, window_tokens: int = WINDOW_TOKENS,
                   step_tokens: int = WINDOW_STEP) -> list[tuple[str, int, int]]:
    """
    Split counsellor utterances into overlapping text windows.
    Returns list of (window_text, start_utt_idx, end_utt_idx).
    """
    # Only use counsellor utterances for coverage
    counsellor_utts = []
    for i, u in enumerate(utterances):
        if isinstance(u, dict):
            spk = u.get("speaker", "")
            txt = u.get("english_text", "") or u.get("native_text", "")
        else:
            spk = getattr(u.speaker, "value", str(u.speaker))
            txt = getattr(u, "english_text", "") or getattr(u, "native_text", "")
        if "counsellor" in spk.lower() or "counselor" in spk.lower():
            counsellor_utts.append((i, txt))

    if not counsellor_utts:
        logger.warning("No counsellor utterances found for semantic alignment")
        return []

    windows = []
    tokens_acc = []
    start_idx  = 0

    for utt_idx, (orig_idx, txt) in enumerate(counsellor_utts):
        words = txt.split()
        tokens_acc.append((orig_idx, words))

        total = sum(len(w) for _, w in tokens_acc)
        if total >= window_tokens:
            window_text = " ".join(w for _, ws in tokens_acc for w in ws)
            windows.append((window_text, start_idx, utt_idx))
            # Step forward
            step_words = 0
            while tokens_acc and step_words < step_tokens:
                step_words += len(tokens_acc[0][1])
                tokens_acc.pop(0)
            start_idx = utt_idx - len(tokens_acc) + 1

    # Flush remaining
    if tokens_acc:
        window_text = " ".join(w for _, ws in tokens_acc for w in ws)
        if window_text.strip():
            windows.append((window_text, start_idx, len(counsellor_utts) - 1))

    # Always have at least one window (full transcript)
    if not windows:
        full = " ".join(t for _, t in counsellor_utts)
        if full.strip():
            windows.append((full, 0, len(counsellor_utts) - 1))

    return windows


# ── Embedding ─────────────────────────────────────────────────────────────────

def _embed_batch(client, texts: list[str]) -> list[list[float] | None]:
    if not texts:
        return []
    try:
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
        return [item.embedding for item in sorted(resp.data, key=lambda x: x.index)]
    except Exception as e:
        logger.warning(f"Embedding error: {e}")
        return [None] * len(texts)


# ── Main aligner ──────────────────────────────────────────────────────────────

def align_transcript_to_deck(
    session_id: str,
    utterances: list,
    deck_id: str,
) -> DeckCoverageReport:
    """
    Core evaluation function.
    Embeds sliding windows of counsellor speech and compares against each
    deck criterion + its paraphrases using cosine similarity.

    Returns DeckCoverageReport with per-criterion results and aggregate score.
    """
    if not settings.openai_api_key:
        logger.warning("No OpenAI key — skipping semantic alignment")
        return _empty_report(deck_id, session_id)

    import openai
    from deck import db as deck_db

    client  = openai.OpenAI(api_key=settings.openai_api_key)
    criteria = deck_db.get_deck_criteria(deck_id, include_disabled=False)

    if not criteria:
        logger.warning(f"No criteria found for deck {deck_id}")
        return _empty_report(deck_id, session_id)

    # Build text windows from counsellor speech
    windows = _build_windows(utterances)
    if not windows:
        return _empty_report(deck_id, session_id)

    logger.info(f"Aligning {len(windows)} windows × {len(criteria)} criteria for {session_id}")

    # Embed all windows in one batch call
    window_texts = [w[0] for w in windows]
    window_embs  = _embed_batch(client, window_texts)

    # Evaluate each criterion
    coverage_results: list[CoverageResult] = []

    for crit in criteria:
        crit_emb    = crit.get("embedding")
        threshold   = crit.get("detection_threshold", 0.50)
        paraphrases = deck_db.get_criterion_paraphrases(crit["criterion_id"])
        para_embs   = [p["embedding"] for p in paraphrases if p.get("embedding")]

        best_score    = 0.0
        best_win_idx  = -1
        best_mode     = "none"

        for w_idx, w_emb in enumerate(window_embs):
            if w_emb is None:
                continue

            # Direct criterion similarity
            if crit_emb:
                direct_score = _cosine(w_emb, crit_emb)
                if direct_score > best_score:
                    best_score   = direct_score
                    best_win_idx = w_idx
                    best_mode    = "direct"

            # Paraphrase similarity
            for p_emb in para_embs:
                p_score = _cosine(w_emb, p_emb)
                if p_score > best_score:
                    best_score   = p_score
                    best_win_idx = w_idx
                    best_mode    = "paraphrase"

        covered = best_score >= threshold

        coverage_results.append(CoverageResult(
            criterion_id    = crit["criterion_id"],
            label           = crit["label"],
            importance      = crit["importance"],
            slide_number    = crit["slide_number"],
            covered         = covered,
            best_similarity = round(best_score, 4),
            best_window_idx = best_win_idx,
            detection_mode  = best_mode if covered else "none",
            threshold       = threshold,
        ))

    # Save to DB
    deck_db.save_coverage_results(session_id, deck_id, [
        {
            "criterion_id":    r.criterion_id,
            "covered":         r.covered,
            "best_similarity": r.best_similarity,
            "best_window_idx": r.best_window_idx,
            "detection_mode":  r.detection_mode,
        }
        for r in coverage_results
    ])
    deck_db.link_call_to_deck(session_id, deck_id)

    return _build_report(deck_id, session_id, coverage_results)


def _build_report(deck_id: str, session_id: str,
                  results: list[CoverageResult]) -> DeckCoverageReport:
    """Aggregate per-criterion results into a coverage score."""
    WEIGHTS = {"must": 3.0, "should": 1.5, "can": 0.5}

    total_weight   = 0.0
    covered_weight = 0.0
    must_covered   = must_total   = 0
    should_covered = should_total = 0
    can_covered    = can_total    = 0
    missed_must    = []
    missed_should  = []

    for r in results:
        w = WEIGHTS.get(r.importance, 1.0)
        total_weight += w
        if r.covered:
            covered_weight += w
        if r.importance == "must":
            must_total += 1
            if r.covered: must_covered += 1
            else: missed_must.append(r.label)
        elif r.importance == "should":
            should_total += 1
            if r.covered: should_covered += 1
            else: missed_should.append(r.label)
        elif r.importance == "can":
            can_total += 1
            if r.covered: can_covered += 1

    score = round((covered_weight / total_weight * 100) if total_weight else 0.0, 1)

    return DeckCoverageReport(
        deck_id        = deck_id,
        session_id     = session_id,
        coverage_score = score,
        must_covered   = must_covered,
        must_total     = must_total,
        should_covered = should_covered,
        should_total   = should_total,
        can_covered    = can_covered,
        can_total      = can_total,
        results        = results,
        missed_must    = missed_must,
        missed_should  = missed_should[:5],
    )


def _empty_report(deck_id: str, session_id: str) -> DeckCoverageReport:
    return DeckCoverageReport(
        deck_id=deck_id, session_id=session_id,
        coverage_score=0.0, must_covered=0, must_total=0,
        should_covered=0, should_total=0, can_covered=0, can_total=0,
    )


def coverage_report_to_dict(report: DeckCoverageReport) -> dict:
    return {
        "deck_id":        report.deck_id,
        "session_id":     report.session_id,
        "coverage_score": report.coverage_score,
        "must_covered":   report.must_covered,
        "must_total":     report.must_total,
        "should_covered": report.should_covered,
        "should_total":   report.should_total,
        "can_covered":    report.can_covered,
        "can_total":      report.can_total,
        "missed_must":    report.missed_must,
        "missed_should":  report.missed_should,
        "results": [
            {
                "criterion_id":   r.criterion_id,
                "label":          r.label,
                "importance":     r.importance,
                "slide_number":   r.slide_number,
                "covered":        r.covered,
                "best_similarity":r.best_similarity,
                "detection_mode": r.detection_mode,
            }
            for r in report.results
        ],
    }
