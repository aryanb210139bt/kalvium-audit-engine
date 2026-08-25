"""
deck/ingestion/schema_generator.py
Uses GPT-4o-mini to extract evaluation criteria from each slide,
then embeds them with text-embedding-3-small.
"""
from __future__ import annotations
import json
import logging
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from config.settings import get_settings
from deck.ingestion.pptx_extractor import RawSlide, extract_pptx
from deck import db

logger = logging.getLogger(__name__)
settings = get_settings()

# Slide categories that rarely generate evaluation criteria
SKIP_CATEGORIES = {"cover", "agenda", "thank_you", "divider", "animation_only"}

SLIDE_CATEGORIES = [
    "intro", "product_overview", "differentiator", "placement",
    "admission", "pricing", "university", "close", "student_life",
    "faq", "testimonial", "comparison", "misc", "cover",
]

SECTION_MAP = {
    "intro":           "opening",
    "product_overview":"product",
    "differentiator":  "product",
    "placement":       "placement",
    "admission":       "admission",
    "pricing":         "fees",
    "university":      "product",
    "close":           "close",
    "faq":             "objections",
    "testimonial":     "trust",
    "comparison":      "trust",
    "student_life":    "product",
    "misc":            "general",
    "cover":           "general",
}


# ── GPT slide analysis ─────────────────────────────────────────────────────────

def _analyse_slide(client, slide: RawSlide, deck_id: str) -> dict | None:
    """Send a slide to GPT-4o-mini and get back structured criteria."""
    if slide.is_empty():
        logger.debug(f"Slide {slide.slide_number}: empty, skipping")
        return None

    prompt = f"""You are a training quality specialist for Indian edtech sales counsellors at Kalvium.
A counsellor presents a slide deck during a video demo call with students and parents.
Your job: extract what a counsellor MUST, SHOULD, or CAN communicate from this slide.

SLIDE {slide.slide_number} — "{slide.title}"
Content:
{slide.body_text[:1200]}
{f'Speaker notes: {slide.speaker_notes[:400]}' if slide.speaker_notes else ''}
{f'Table data: {" | ".join(slide.table_cells[:20])}' if slide.table_cells else ''}

Return ONLY valid JSON with this exact structure:
{{
  "slide_category": "one of: intro|product_overview|differentiator|placement|admission|pricing|university|close|faq|testimonial|comparison|student_life|misc|cover",
  "skip": false,
  "skip_reason": "optional: only if slide has no audit value (e.g. purely decorative)",
  "criteria": [
    {{
      "label": "Short label (max 10 words)",
      "explanation": "What the counsellor must communicate. Be specific to this slide's content. 1-2 sentences.",
      "importance": "must|should|can",
      "paraphrases": [
        "Natural conversational way a counsellor might say this #1",
        "Natural conversational way a counsellor might say this #2",
        "Natural conversational way a counsellor might say this #3"
      ]
    }}
  ]
}}

Rules:
- "must" = critical information counsellor cannot skip (max 2-3 per slide)
- "should" = important but call can still succeed without it
- "can" = nice-to-have, contextual
- Keep criteria focused on THIS slide's content, not generic counselling skills
- If slide is a cover/divider/animation with no real content, set skip=true
- Indian education context: KNET is the entrance test, Kalvium is B.Tech CSE
- Paraphrases should sound natural and conversational, not formal"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1200,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content
        data = json.loads(raw)
        return data
    except Exception as e:
        logger.warning(f"Slide {slide.slide_number} GPT analysis failed: {e}")
        return None


# ── Embedding ─────────────────────────────────────────────────────────────────

def _embed_texts(client, texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using text-embedding-3-small."""
    if not texts:
        return []
    try:
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
        )
        return [item.embedding for item in sorted(resp.data, key=lambda x: x.index)]
    except Exception as e:
        logger.warning(f"Embedding failed: {e}")
        return [None] * len(texts)


# ── Main generator ─────────────────────────────────────────────────────────────

def _extract_holistic_criteria(client, slides: list, deck_name: str) -> list[dict]:
    """
    Single GPT-4o call on the full deck summary →
    returns 12-15 holistic call-level criteria a counsellor must cover.
    """
    # Build a compact deck summary: slide number + title + first 120 chars of body
    slide_lines = []
    for s in slides:
        if s.is_empty():
            continue
        body_preview = (s.body_text or "")[:120].replace("\n", " ").strip()
        line = f"Slide {s.slide_number}: {s.title or '(no title)'}"
        if body_preview:
            line += f" — {body_preview}"
        slide_lines.append(line)

    deck_summary = "\n".join(slide_lines[:80])   # cap at 80 slides to stay in token budget

    prompt = f"""You are a training quality analyst for Kalvium, an Indian edtech B.Tech CSE programme.
A sales counsellor presents this deck during a 30-60 minute demo call with students and parents.

DECK: "{deck_name}"
SLIDE SUMMARY:
{deck_summary}

Your task: identify exactly 12-15 KEY TOPICS that a counsellor MUST or SHOULD cover during this demo call.
These must be:
- High-level call topics (not slide-specific micro-details)
- Things a counsellor would naturally mention in spoken conversation
- Broad enough to match natural speech ("our fee is X lakhs" matches "tuition fee information")
- Ordered by importance (most critical first)

Return ONLY valid JSON:
{{
  "criteria": [
    {{
      "label": "Short topic name (max 8 words)",
      "explanation": "What the counsellor should communicate about this topic. 1-2 sentences. Written as spoken content, not as a slide description.",
      "importance": "must or should",
      "paraphrases": [
        "A natural way a counsellor might say this in conversation",
        "Another natural phrasing",
        "A third natural phrasing"
      ]
    }}
  ]
}}

Rules:
- Maximum 15 criteria total
- At most 8 can be "must" — only truly critical topics
- Remaining are "should"
- No "can" importance — keep list tight
- Paraphrases must sound like real spoken counsellor language, not formal text
- Cover the full arc of a demo call: intro → programme → differentiators → placement → fees → admission → close"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=2500,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        criteria = data.get("criteria", [])
        logger.info(f"Holistic extraction: {len(criteria)} criteria")
        return criteria[:15]   # hard cap
    except Exception as e:
        logger.error(f"Holistic criteria extraction failed: {e}")
        return []


def generate_schema_from_pptx(
    pptx_path: str | Path,
    deck_name: str,
    created_by: str = "system",
) -> dict:
    """
    Full pipeline:
      1. Extract slides from PPTX
      2. Single GPT-4o call on full deck → 12-15 holistic call criteria
      3. Embed all criteria + paraphrases
      4. Save to deck DB
      5. Return summary dict
    """
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY not set in .env")

    import openai
    client = openai.OpenAI(api_key=settings.openai_api_key)

    pptx_path = Path(pptx_path)
    slides = extract_pptx(pptx_path)

    slug    = re.sub(r"[^a-z0-9]+", "-", deck_name.lower())[:40]
    deck_id = f"{slug}-{uuid.uuid4().hex[:8]}"
    now     = datetime.utcnow().isoformat()

    logger.info(f"Generating schema for '{deck_name}' ({len(slides)} slides) → {deck_id}")

    deck_meta = {
        "deck_id":          deck_id,
        "deck_slug":        slug,
        "deck_name":        deck_name,
        "status":           "draft",
        "source_filename":  pptx_path.name,
        "slide_count":      len(slides),
        "total_criteria":   0,
        "created_at":       now,
        "created_by":       created_by,
        "scoring_weights":  {"behavioral_layer": 0.55, "deck_coverage_layer": 0.45},
        "changelog": [{"version": "1.0.0", "date": now[:10],
                       "author": created_by, "changes": "Initial auto-extraction"}],
    }
    db.save_deck_schema(deck_meta)

    # Save slide records (lightweight — no per-slide GPT)
    for slide in slides:
        db.save_slide({
            "deck_id":       deck_id,
            "slide_number":  slide.slide_number,
            "title":         slide.title,
            "raw_text":      slide.full_text[:2000],
            "speaker_notes": slide.speaker_notes[:500],
            "slide_category":"misc",
            "section_id":    "general",
        })

    # ── Phase 1: single holistic GPT call → 12-15 criteria ───────────────────
    raw_criteria = _extract_holistic_criteria(client, slides, deck_name)
    if not raw_criteria:
        raise RuntimeError("GPT failed to extract criteria from deck")

    all_criteria_for_embedding: list[tuple[str, str, list[str]]] = []

    for idx, c in enumerate(raw_criteria):
        criterion_id = f"{deck_id}_c{idx+1}"
        paraphrases  = c.get("paraphrases", [])
        db.save_criterion({
            "criterion_id":        criterion_id,
            "deck_id":             deck_id,
            "slide_number":        0,          # deck-level, not slide-specific
            "label":               c.get("label", "")[:120],
            "explanation":         c.get("explanation", ""),
            "importance":          c.get("importance", "should"),
            "slide_section":       "general",
            "embedding":           None,
            "detection_threshold": 0.50,       # tuned for natural speech
            "disabled":            False,
            "display_order":       idx,
        })
        all_criteria_for_embedding.append((criterion_id, c.get("explanation", ""), paraphrases))

    total_criteria = len(all_criteria_for_embedding)
    logger.info(f"Extracted {total_criteria} holistic criteria")

    # ── Phase 2: batch embed criteria + paraphrases ───────────────────────────
    logger.info(f"Embedding {total_criteria} criteria + paraphrases…")
    explanation_texts = [exp for _, exp, _ in all_criteria_for_embedding]
    explanation_embs  = _embed_texts(client, explanation_texts)

    import json as _json
    conn = db._conn()
    for i, (crit_id, _exp, paraphrases) in enumerate(all_criteria_for_embedding):
        emb = explanation_embs[i] if i < len(explanation_embs) else None
        if emb:
            conn.execute(
                "UPDATE deck_criteria SET embedding_json=? WHERE criterion_id=?",
                (_json.dumps(emb), crit_id)
            )
        conn.commit()
        if paraphrases:
            para_embs = _embed_texts(client, paraphrases)
            for txt, pemb in zip(paraphrases, para_embs):
                db.save_paraphrase(crit_id, txt, pemb)

    # Update count
    conn.execute(
        "UPDATE deck_schemas SET total_criteria=? WHERE deck_id=?",
        (total_criteria, deck_id)
    )
    conn.commit()

    logger.info(f"✅ Schema done: {deck_id} — {total_criteria} criteria at threshold=0.50")

    return {
        "deck_id":        deck_id,
        "deck_name":      deck_name,
        "slide_count":    len(slides),
        "total_criteria": total_criteria,
        "status":         "draft",
        "created_at":     now,
    }
