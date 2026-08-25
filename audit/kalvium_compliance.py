"""
audit/kalvium_compliance.py
Layer 1 — Kalvium Registration Compliance Analyzer.

Checks whether the associate followed all 10 steps of the Kalvium
booking call process using both rule signals and GPT-4o verification.
"""
from __future__ import annotations
import json
import logging
import re

from openai import OpenAI

from config.models import Utterance
from config.kalvium_models import KalviumComplianceResult
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# ── Rule-based keyword signals per step ───────────────────────────────────────

_STEP_SIGNALS: dict[str, list[str]] = {
    "introduction_done": [
        r"\bmy name is\b", r"\bI am\b.*\bkalvium\b", r"\bcalling from kalvium\b",
        r"\bkalvium team\b", r"\bhi.*I am\b", r"\bhello.*name\b",
        r"\bnama.*kalvium\b", r"\bmain.*kalvium\b", r"\bme.*kalvium\b",
    ],
    "qualification_completed": [
        r"\bpcm\b", r"\bpcmb\b", r"\bclass 12\b", r"\b12th\b", r"\bdropper\b",
        r"\bphysics.*chemistry.*math\b", r"\bkaunsi class\b", r"\bwhich class\b",
        r"\bstream\b.*\bscience\b", r"\bkaunsa subject\b",
    ],
    "ai_discussion_done": [
        r"\bai\b.*\bjob\b", r"\bartificial intelligence\b", r"\bautomation\b",
        r"\bjob market\b", r"\bfuture.*job\b", r"\bai.*replace\b",
        r"\btechnology.*impact\b", r"\bnaukri\b.*\bai\b", r"\bai.*career\b",
    ],
    "kalvium_differentiation_explained": [
        r"\bkalvium\b.*\bunique\b", r"\bindustry.*curriculum\b",
        r"\bwork.*integrat\b", r"\bplacement\b", r"\breal.*project\b",
        r"\btraditional.*college\b", r"\bunlike.*other\b", r"\bbetter.*than\b",
        r"\bdifference\b.*\bkalvium\b",
    ],
    "webinar_pitched": [
        r"\bwebinar\b", r"\bdemo\b.*\bsession\b", r"\bonline.*session\b",
        r"\bfree.*session\b", r"\bjoin.*session\b", r"\bregister\b",
        r"\bfree.*webinar\b", r"\bsession.*register\b",
    ],
    "email_collected": [
        r"\bemail\b", r"\bemail id\b", r"\byour email\b",
        r"\bapka email\b", r"\bmail id\b", r"\bgmail\b",
        r"\@", r"\baddress.*email\b",
    ],
    "email_verified": [
        r"\bspell.*email\b", r"\bconfirm.*email\b", r"\brepeat.*email\b",
        r"\bemail.*correct\b", r"\bthat.*right.*email\b",
        r"\bdobara.*email\b", r"\bsahi hai.*email\b",
    ],
    "attendance_commitment_obtained": [
        r"\bwill you.*attend\b", r"\bconfirm.*attendance\b",
        r"\bpakka.*aayenge\b", r"\bpakka.*attend\b", r"\bpromise.*attend\b",
        r"\bsure.*join\b", r"\bwill you join\b", r"\bconfirm.*join\b",
    ],
    "registration_initiated": [
        r"\bregistered\b", r"\bregistration.*done\b", r"\bbooked\b",
        r"\bregister.*kar diya\b", r"\bseat.*book\b", r"\bapka seat\b",
    ],
    "email_receipt_confirmed": [
        r"\bcheck.*email\b", r"\bmail.*send\b", r"\bconfirmation.*mail\b",
        r"\binbox.*check\b", r"\bmail.*aayega\b", r"\bteams.*link\b",
    ],
}


def _rule_check(utterances: list[Utterance]) -> dict[str, bool]:
    """Fast regex pass over all utterance text to pre-score compliance steps."""
    full_text = " ".join(
        u.english_text.lower() + " " + u.native_text.lower()
        for u in utterances
    )
    result = {}
    for step, patterns in _STEP_SIGNALS.items():
        result[step] = any(re.search(p, full_text, re.IGNORECASE) for p in patterns)
    return result


def _build_compliance_prompt(transcript: str) -> str:
    return f"""You are an expert sales quality auditor for Kalvium, an ed-tech company.

Kalvium associates call students (Class 12 PCM/PCMB, droppers, referred students)
to register them for free webinar/demo sessions about Kalvium's B.Tech program.

EXPECTED ASSOCIATE WORKFLOW (10 steps):
1. Self-introduction (name + Kalvium mention)
2. Prospect qualification (PCM/PCMB/Class 12/dropper status)
3. AI impact on jobs discussion
4. Kalvium differentiation explanation (industry curriculum, placement, vs traditional college)
5. Webinar/demo pitch and value proposition
6. Registration process initiation
7. Email ID collection from prospect
8. Email repetition/spelling/verification by associate
9. Email receipt confirmation (check your inbox, Teams link)
10. Webinar attendance commitment obtained

TRANSCRIPT:
{transcript}

TASK:
Analyze whether the associate completed each step. Be strict. Partial mentions do not count.
For email verification: the associate must have actually REPEATED or SPELLED OUT the email.
Return ONLY valid JSON — no explanation outside the JSON block:

{{
  "compliance_score": <0-100 integer>,
  "steps_completed": ["list of step names completed"],
  "missing_steps": ["list of step names skipped"],
  "introduction_done": <true/false>,
  "qualification_completed": <true/false>,
  "ai_discussion_done": <true/false>,
  "kalvium_differentiation_explained": <true/false>,
  "webinar_pitched": <true/false>,
  "registration_initiated": <true/false>,
  "email_collected": <true/false>,
  "email_verified": <true/false>,
  "email_receipt_confirmed": <true/false>,
  "attendance_commitment_obtained": <true/false>,
  "compliance_observations": ["specific observations"],
  "compliance_notes": "<one paragraph summary>"
}}"""


class KalviumComplianceAnalyzer:
    """
    Layer 1: Sales compliance analysis for Kalvium booking calls.
    Uses rule signals for speed, then GPT-4o for semantic verification.
    """

    def __init__(self):
        if settings.openai_api_key:
            self._client = OpenAI(api_key=settings.openai_api_key)
        else:
            self._client = None

    def analyze(
        self,
        utterances: list[Utterance],
        full_transcript: str,
    ) -> KalviumComplianceResult:
        rule_hits = _rule_check(utterances)

        if not self._client:
            logger.warning("No OpenAI key — using rule-only compliance scoring")
            return self._from_rule_hits(rule_hits)

        prompt = _build_compliance_prompt(full_transcript[:12000])
        try:
            resp = self._client.chat.completions.create(
                model=settings.openai_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=900,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or "{}"
            data = json.loads(raw)
            return self._parse_llm_output(data, rule_hits)
        except Exception as exc:
            logger.error(f"Compliance LLM call failed: {exc}")
            return self._from_rule_hits(rule_hits)

    # ── parsers ───────────────────────────────────────────────────────────────

    def _parse_llm_output(
        self, data: dict, rule_hits: dict[str, bool]
    ) -> KalviumComplianceResult:
        # Rule hits act as a floor — if rule says true, override LLM false
        def _bool(key: str) -> bool:
            return bool(data.get(key, False)) or rule_hits.get(key, False)

        steps_done = len([k for k in _STEP_SIGNALS if _bool(k)])
        total = len(_STEP_SIGNALS)
        # Use LLM score but floor it to rule-based completion rate
        rule_pct = round(steps_done / total * 100)
        llm_score = float(data.get("compliance_score", rule_pct))
        compliance_score = max(rule_pct * 0.5, llm_score)  # blend

        return KalviumComplianceResult(
            compliance_score=min(100, round(compliance_score, 1)),
            steps_completed=data.get("steps_completed", []),
            missing_steps=data.get("missing_steps", []),
            introduction_done=_bool("introduction_done"),
            qualification_completed=_bool("qualification_completed"),
            ai_discussion_done=_bool("ai_discussion_done"),
            kalvium_differentiation_explained=_bool("kalvium_differentiation_explained"),
            webinar_pitched=_bool("webinar_pitched"),
            email_collected=_bool("email_collected"),
            email_verified=_bool("email_verified"),
            attendance_commitment_obtained=_bool("attendance_commitment_obtained"),
            registration_initiated=_bool("registration_initiated"),
            email_receipt_confirmed=_bool("email_receipt_confirmed"),
            compliance_observations=data.get("compliance_observations", []),
            compliance_notes=data.get("compliance_notes", ""),
        )

    def _from_rule_hits(self, rule_hits: dict[str, bool]) -> KalviumComplianceResult:
        steps_done = [k for k, v in rule_hits.items() if v]
        missing = [k for k, v in rule_hits.items() if not v]
        score = round(len(steps_done) / len(_STEP_SIGNALS) * 100, 1)
        return KalviumComplianceResult(
            compliance_score=score,
            steps_completed=steps_done,
            missing_steps=missing,
            **{k: v for k, v in rule_hits.items()},
        )
