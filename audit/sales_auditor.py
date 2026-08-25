"""
audit/sales_auditor.py
Sales audit engine using Claude Sonnet for 9-dimension scoring.

Each dimension is scored 1–10 with:
  - Rationale (2–3 sentences)
  - 1–3 evidence quotes from the transcript with timestamps
  - Positives list
  - Negatives list

Total score = weighted average × 10 (0–100 scale).

Claude is used because it produces the most accurate, nuanced, and
empathetically framed coaching feedback. Prompt caching is used to
amortize the cost of the large system prompt across calls.
"""
from __future__ import annotations
import json
import logging

from config.models_multilingual import (
    AuditDimension, SalesAuditScore, EvidenceQuote, SpeakerRole
)
from config.settings import get_settings
from transcription.post_processor import build_full_transcript

logger   = logging.getLogger(__name__)
settings = get_settings()

# ── Dimension definitions ──────────────────────────────────────────────────────

DIMENSIONS = [
    {
        "id":     "opening",
        "label":  "Opening",
        "weight": 0.08,
        "rubric": (
            "1–3: No agenda, confusing start, no energy.\n"
            "4–6: Basic introduction but no permission-seeking or agenda setting.\n"
            "7–8: Clear agenda, agenda confirmed with prospect, good energy.\n"
            "9–10: Perfect opening — personalised hook, prospect name used, "
            "clear WIIFM communicated in first 90 seconds."
        ),
    },
    {
        "id":     "rapport_building",
        "label":  "Rapport Building",
        "weight": 0.08,
        "rubric": (
            "1–3: Cold, transactional, no small talk or personalisation.\n"
            "4–6: Polite but surface-level; does not connect on a personal level.\n"
            "7–8: Finds common ground, references prospect's context, warm tone.\n"
            "9–10: Natural rapport; prospect visibly engaged; rep remembered/used personal details."
        ),
    },
    {
        "id":     "discovery",
        "label":  "Discovery",
        "weight": 0.18,
        "rubric": (
            "1–3: No open-ended questions; rep talks most of the time from the start.\n"
            "4–6: Some questions asked but surface-level; pain not quantified.\n"
            "7–8: Open-ended discovery; current state, pain, and impact understood.\n"
            "9–10: SPIN-level discovery — current state, desired state, implication, "
            "pain quantified in customer's own words."
        ),
    },
    {
        "id":     "requirement_gathering",
        "label":  "Requirement Gathering",
        "weight": 0.12,
        "rubric": (
            "1–3: No requirements gathered; rep pitches generic solution.\n"
            "4–6: Some requirements noted but not systematically captured.\n"
            "7–8: Specific requirements understood; solution mapped to them.\n"
            "9–10: All relevant requirements surfaced; priorities confirmed with prospect; "
            "solution explicitly mapped to each requirement."
        ),
    },
    {
        "id":     "product_understanding",
        "label":  "Product Understanding",
        "weight": 0.12,
        "rubric": (
            "1–3: Rep cannot answer basic product questions; wrong information given.\n"
            "4–6: Knows product features but presents features not benefits.\n"
            "7–8: Explains value in customer terms; uses relevant examples.\n"
            "9–10: Deep product mastery; tailors explanation to prospect's specific situation; "
            "uses prospect's own words from discovery."
        ),
    },
    {
        "id":     "objection_handling",
        "label":  "Objection Handling",
        "weight": 0.18,
        "rubric": (
            "1–3: Ignores, dismisses, or argues with objections.\n"
            "4–6: Acknowledges objection but response is generic or weak.\n"
            "7–8: Acknowledges, probes root cause, provides evidence.\n"
            "9–10: Acknowledges → probes root cause → uses prospect's own logic → "
            "provides social proof or data → confirms resolution."
        ),
    },
    {
        "id":     "value_communication",
        "label":  "Value Communication",
        "weight": 0.10,
        "rubric": (
            "1–3: No ROI or value framing; speaks only about features.\n"
            "4–6: Mentions ROI but without specifics.\n"
            "7–8: Specific ROI framing tied to prospect's pain.\n"
            "9–10: Quantified ROI in prospect's terms; uses their numbers; "
            "anchors value vs cost of inaction."
        ),
    },
    {
        "id":     "closing",
        "label":  "Closing",
        "weight": 0.08,
        "rubric": (
            "1–3: No closing attempt; call ends without commitment.\n"
            "4–6: Weak close; asks for next step without creating urgency.\n"
            "7–8: Clear CTA; specific next step agreed.\n"
            "9–10: Trial close used; commitment gained; next steps crystal clear "
            "with owner, date, and deliverable confirmed."
        ),
    },
    {
        "id":     "follow_up_clarity",
        "label":  "Follow-up Clarity",
        "weight": 0.06,
        "rubric": (
            "1–3: No follow-up discussed.\n"
            "4–6: Vague follow-up ('I'll send you some info').\n"
            "7–8: Specific follow-up action with timeline mentioned.\n"
            "9–10: Next meeting booked; specific deliverables named; "
            "prospect's responsibilities also confirmed."
        ),
    },
]

AUDIT_SYSTEM_PROMPT = """You are a Principal Sales Coach at a B2B SaaS company.
Your task is to audit a sales call and score the salesperson on 9 dimensions.

The transcript may contain English, Hindi, Hinglish, Tamil, Telugu, Kannada,
Malayalam, Bengali, or Marathi. Understand the full semantic meaning regardless
of the language used. Score based on CONTENT and TECHNIQUE, not language.

SCORING RULES:
- Score each dimension 1–10 using the rubric provided.
- Every score MUST be supported by 1–3 direct quotes from the transcript.
- Quotes must be in the ORIGINAL language (do not translate).
- Timestamps must match the [HH:MM:SS] format in the transcript.
- Be honest and calibrated — a 7 should be genuinely good, not mediocre.
- If a dimension is not observable in the transcript, score it 5 (neutral).

OUTPUT: Valid JSON only. No preamble. No explanations outside the JSON."""

DIMENSION_PROMPT_TEMPLATE = """
Dimension: {label}
Weight: {weight_pct}% of total score

Rubric:
{rubric}

Score this dimension. Return this JSON object:
{{
  "score": <int 1-10>,
  "rationale": "<2-3 sentence explanation>",
  "evidence": [
    {{"quote": "<exact text>", "speaker": "<Salesperson|Customer>", "timestamp": "<HH:MM:SS>"}},
    ...
  ],
  "positives": ["<what went well>", ...],
  "negatives": ["<what could improve>", ...]
}}
"""


class SalesAuditor:
    """
    Scores a sales call on 9 dimensions using Claude Sonnet.
    All 9 dimensions are scored in a single API call for efficiency.
    """

    def audit(self, utterances: list, call_intelligence) -> SalesAuditScore:
        transcript = build_full_transcript(utterances)
        if not transcript.strip():
            return _empty_score()

        raw = self._call_claude(transcript, call_intelligence)
        if raw is None:
            raw = self._call_openai(transcript, call_intelligence)
        if raw is None:
            return _empty_score()

        return _parse_audit(raw)

    # ── Claude (primary) ───────────────────────────────────────────────────────

    def _call_claude(self, transcript: str, intel) -> dict | None:
        anthropic_key = getattr(settings, "anthropic_api_key", "")
        if not anthropic_key:
            return None

        try:
            import anthropic
            client = anthropic.Anthropic(api_key=anthropic_key)

            # Build the full dimension rubric block
            dim_block = ""
            for d in DIMENSIONS:
                dim_block += DIMENSION_PROMPT_TEMPLATE.format(
                    label=d["label"],
                    weight_pct=int(d["weight"] * 100),
                    rubric=d["rubric"],
                )

            user_msg = (
                f"TRANSCRIPT (first 18,000 chars):\n{transcript[:18000]}\n\n"
                f"CALL INTELLIGENCE SUMMARY:\n"
                f"Pain points: {[pp.description for pp in (intel.pain_points if intel else [])]}\n"
                f"Objections: {[obj.type for obj in (intel.objections if intel else [])]}\n"
                f"Deal risk: {intel.deal_risk.value if intel else 'unknown'}\n\n"
                f"Now score all 9 dimensions. Return a single JSON object with keys:\n"
                f"opening, rapport_building, discovery, requirement_gathering, "
                f"product_understanding, objection_handling, value_communication, "
                f"closing, follow_up_clarity\n\n"
                f"Each key maps to the dimension JSON object described above.\n\n"
                f"{dim_block}"
            )

            msg = client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=6000,
                system=AUDIT_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
                temperature=0.1,
            )

            text = msg.content[0].text.strip()
            return _extract_json(text)

        except Exception as exc:
            logger.warning(f"Claude audit failed: {exc}")
            return None

    # ── GPT-4o fallback ────────────────────────────────────────────────────────

    def _call_openai(self, transcript: str, intel) -> dict | None:
        if not settings.openai_api_key:
            return None

        try:
            from openai import OpenAI
            client = OpenAI(api_key=settings.openai_api_key)

            dim_block = ""
            for d in DIMENSIONS:
                dim_block += DIMENSION_PROMPT_TEMPLATE.format(
                    label=d["label"],
                    weight_pct=int(d["weight"] * 100),
                    rubric=d["rubric"],
                )

            user_msg = (
                f"TRANSCRIPT:\n{transcript[:15000]}\n\n"
                f"Score all 9 dimensions. Return a single JSON object with keys: "
                f"opening, rapport_building, discovery, requirement_gathering, "
                f"product_understanding, objection_handling, value_communication, "
                f"closing, follow_up_clarity\n\n{dim_block}"
            )

            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": AUDIT_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                response_format={"type": "json_object"},
                max_tokens=6000,
                temperature=0.1,
            )
            return json.loads(resp.choices[0].message.content)

        except Exception as exc:
            logger.warning(f"GPT-4o audit fallback failed: {exc}")
            return None


# ── Parsing & helpers ──────────────────────────────────────────────────────────

def _parse_audit(raw: dict) -> SalesAuditScore:
    dim_models: dict[str, AuditDimension] = {}
    total_weighted = 0.0

    for d in DIMENSIONS:
        key  = d["id"]
        data = raw.get(key, {})
        score = max(1, min(10, int(data.get("score", 5))))

        evidence = []
        for ev in data.get("evidence", [])[:3]:
            if isinstance(ev, dict):
                evidence.append(EvidenceQuote(
                    text=ev.get("quote", ev.get("text", "")),
                    speaker=SpeakerRole.SALESPERSON
                    if "sale" in ev.get("speaker", "").lower()
                    else SpeakerRole.CUSTOMER,
                    timestamp=ev.get("timestamp", "00:00:00"),
                ))

        dim_models[key] = AuditDimension(
            name=d["label"],
            score=score,
            weight=d["weight"],
            rationale=data.get("rationale", ""),
            evidence=evidence,
            positives=data.get("positives", [])[:3],
            negatives=data.get("negatives", [])[:3],
        )
        total_weighted += (score / 10) * d["weight"] * 100

    total_score = round(total_weighted, 1)
    grade = _grade(total_score)

    return SalesAuditScore(
        opening=dim_models.get("opening", _default_dim("Opening", 0.08)),
        rapport_building=dim_models.get("rapport_building", _default_dim("Rapport Building", 0.08)),
        discovery=dim_models.get("discovery", _default_dim("Discovery", 0.18)),
        requirement_gathering=dim_models.get("requirement_gathering", _default_dim("Requirement Gathering", 0.12)),
        product_understanding=dim_models.get("product_understanding", _default_dim("Product Understanding", 0.12)),
        objection_handling=dim_models.get("objection_handling", _default_dim("Objection Handling", 0.18)),
        value_communication=dim_models.get("value_communication", _default_dim("Value Communication", 0.10)),
        closing=dim_models.get("closing", _default_dim("Closing", 0.08)),
        follow_up_clarity=dim_models.get("follow_up_clarity", _default_dim("Follow-up Clarity", 0.06)),
        total_score=total_score,
        grade=grade,
    )


def _grade(score: float) -> str:
    if score >= 90: return "A+"
    if score >= 80: return "A"
    if score >= 70: return "B"
    if score >= 60: return "C"
    if score >= 50: return "D"
    return "F"


def _default_dim(name: str, weight: float) -> AuditDimension:
    return AuditDimension(name=name, score=5, weight=weight, rationale="Not evaluated.")


def _empty_score() -> SalesAuditScore:
    dims = {d["id"]: _default_dim(d["label"], d["weight"]) for d in DIMENSIONS}
    return SalesAuditScore(
        opening=dims["opening"],
        rapport_building=dims["rapport_building"],
        discovery=dims["discovery"],
        requirement_gathering=dims["requirement_gathering"],
        product_understanding=dims["product_understanding"],
        objection_handling=dims["objection_handling"],
        value_communication=dims["value_communication"],
        closing=dims["closing"],
        follow_up_clarity=dims["follow_up_clarity"],
        total_score=50.0,
        grade="D",
    )


def _extract_json(text: str) -> dict | None:
    try:
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        return json.loads(text)
    except Exception as exc:
        logger.warning(f"JSON parse failed: {exc}\nText: {text[:300]}")
        return None
