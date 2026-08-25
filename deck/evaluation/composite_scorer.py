"""
deck/evaluation/composite_scorer.py
Blends the static behavioral score (Layer 1) with the dynamic deck
coverage score (Layer 2) into a single composite score.

If no deck is linked to a session, the composite score = behavioral score
(full backward compatibility).
"""
from __future__ import annotations
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CompositeScore:
    overall:               float
    grade:                 str
    behavioral_score:      float
    behavioral_component:  float   # behavioral_score × w_b
    deck_coverage_score:   float
    deck_component:        float   # deck_coverage_score × w_d
    deck_id:               str | None
    weights_used:          dict


def _grade(score: float) -> str:
    if score >= 90: return "A+"
    if score >= 80: return "A"
    if score >= 70: return "B"
    if score >= 60: return "C"
    if score >= 50: return "D"
    return "F"


def compute_composite(
    behavioral_score: float,
    deck_coverage_score: float | None,
    deck_id: str | None,
    scoring_weights: dict | None = None,
) -> CompositeScore:
    """
    Merge Layer 1 (behavioral) and Layer 2 (deck coverage) scores.

    If deck_coverage_score is None (no deck assigned), returns behavioral
    score unchanged — existing sessions are unaffected.
    """
    if deck_coverage_score is None or deck_id is None:
        # Pure behavioral mode — backward compatible
        return CompositeScore(
            overall               = round(min(100.0, max(0.0, behavioral_score)), 1),
            grade                 = _grade(behavioral_score),
            behavioral_score      = behavioral_score,
            behavioral_component  = behavioral_score,
            deck_coverage_score   = 0.0,
            deck_component        = 0.0,
            deck_id               = None,
            weights_used          = {"behavioral_layer": 1.0, "deck_coverage_layer": 0.0},
        )

    w = scoring_weights or {"behavioral_layer": 0.55, "deck_coverage_layer": 0.45}
    w_b = float(w.get("behavioral_layer", 0.55))
    w_d = float(w.get("deck_coverage_layer", 0.45))

    # Normalise weights in case they don't sum to 1
    total_w = w_b + w_d
    if total_w > 0:
        w_b /= total_w
        w_d /= total_w

    behavioral_component = behavioral_score * w_b
    deck_component       = deck_coverage_score * w_d
    overall              = round(min(100.0, max(0.0, behavioral_component + deck_component)), 1)

    logger.info(
        f"Composite: behavioral={behavioral_score:.1f}×{w_b:.2f} + "
        f"deck={deck_coverage_score:.1f}×{w_d:.2f} = {overall}/100"
    )

    return CompositeScore(
        overall               = overall,
        grade                 = _grade(overall),
        behavioral_score      = behavioral_score,
        behavioral_component  = round(behavioral_component, 1),
        deck_coverage_score   = deck_coverage_score,
        deck_component        = round(deck_component, 1),
        deck_id               = deck_id,
        weights_used          = {"behavioral_layer": round(w_b, 3),
                                 "deck_coverage_layer": round(w_d, 3)},
    )
