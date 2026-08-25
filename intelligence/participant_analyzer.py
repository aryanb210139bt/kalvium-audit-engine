"""
intelligence/participant_analyzer.py
Transcript-based participant intelligence: per-role engagement scoring,
attendance detection, and GPT-generated audit-impact narrative.

Works purely from utterances — no video, no extra models.
"""
from __future__ import annotations
import logging
import re
from typing import Optional

from config.models import Speaker, Utterance
from config.settings import get_settings

logger   = logging.getLogger(__name__)
settings = get_settings()

# ── Question-detection heuristic ─────────────────────────────────────────────
_QUESTION_STARTERS = re.compile(
    r"^\s*(what|how|why|when|where|which|who|whose|whom|is|are|was|were|"
    r"will|would|can|could|do|does|did|should|may|might|shall|have|has|had|"
    r"isn'?t|aren'?t|wasn'?t|weren'?t|won'?t|don'?t|doesn'?t|didn'?t)\b",
    re.IGNORECASE,
)

# ── Parent objection keywords ─────────────────────────────────────────────────
_OBJECTION_PATTERNS = re.compile(
    r"\b(fee|fees|cost|costly|expensive|afford|price|money|lakhs?|rupees?|"
    r"guarantee|placement|package|return|roi|why so much|other college|"
    r"better option|think about it|need time|will discuss|not sure|"
    r"compare|traditional|engineering|btech|b\.tech)\b",
    re.IGNORECASE,
)

# ── Decision / commitment keywords (parent) ───────────────────────────────────
_DECISION_PATTERNS = re.compile(
    r"\b(okay|ok|alright|fine|agree|interested|definitely|sure|when can|"
    r"when does|admission|join|enroll|proceed|go ahead|let'?s do|confirm)\b",
    re.IGNORECASE,
)


def _word_count(text: str) -> int:
    return len(text.split())


def _is_question(text: str) -> bool:
    return text.strip().endswith("?") or bool(_QUESTION_STARTERS.match(text.strip()))


def _is_objection(text: str) -> bool:
    return bool(_OBJECTION_PATTERNS.search(text))


def _is_decision(text: str) -> bool:
    return bool(_DECISION_PATTERNS.search(text))


def _clamp(v: float, lo: float = 0.0, hi: float = 10.0) -> float:
    return max(lo, min(hi, v))


class ParticipantAnalyzer:

    def analyze(self, utterances: list[Utterance], duration_seconds: float) -> dict:
        """
        Returns a dict matching ParticipantIntelligence schema.
        Returns None-equivalent empty dict on failure (caller treats as non-fatal).
        """
        if not utterances:
            return self._empty_result()

        # ── 1. Aggregate per-speaker stats ───────────────────────────────────
        stats: dict[Speaker, dict] = {}
        for spk in Speaker:
            stats[spk] = {
                "word_count": 0, "turn_count": 0,
                "questions": 0, "objections": 0, "decisions": 0,
                "first_time": None, "last_time": None,
            }

        prev_speaker: Optional[Speaker] = None
        for utt in utterances:
            spk = utt.speaker
            s   = stats[spk]
            txt = utt.english_text or utt.native_text or ""

            s["word_count"] += _word_count(txt)
            if spk != prev_speaker:
                s["turn_count"] += 1
            if _is_question(txt):
                s["questions"] += 1
            if spk == Speaker.PARENT and _is_objection(txt):
                s["objections"] += 1
            if spk == Speaker.PARENT and _is_decision(txt):
                s["decisions"] += 1
            if s["first_time"] is None:
                s["first_time"] = utt.start_time
            s["last_time"] = utt.end_time
            prev_speaker = spk

        total_words = sum(s["word_count"] for s in stats.values()) or 1
        total_turns = sum(s["turn_count"] for s in stats.values()) or 1
        total_questions = sum(s["questions"] for s in stats.values()) or 1

        # ── 2. Attendance percentages ────────────────────────────────────────
        def _att_pct(spk: Speaker) -> float:
            s = stats[spk]
            if s["first_time"] is None:
                return 0.0
            if duration_seconds <= 0:
                return 100.0
            present_span = (s["last_time"] or 0) - (s["first_time"] or 0)
            return round(min(present_span / duration_seconds * 100, 100.0), 1)

        student_att = _att_pct(Speaker.STUDENT)
        parent_att  = _att_pct(Speaker.PARENT)
        student_present = stats[Speaker.STUDENT]["word_count"] > 0
        parent_present  = stats[Speaker.PARENT]["word_count"] > 0

        if not student_present:
            quality = "counsellor_only"
        elif not parent_present:
            quality = "student_only"
        elif parent_att >= 85 and student_att >= 85:
            quality = "full"
        else:
            quality = "partial"

        # ── 3. Scores ─────────────────────────────────────────────────────────
        ctr = stats[Speaker.COUNSELLOR]["word_count"] / total_words

        # Student Engagement Score (0–10)
        st = stats[Speaker.STUDENT]
        ses_raw = (
            0.25 * (st["word_count"] / total_words) +
            0.25 * (st["questions"] / max(st["turn_count"], 1)) * 5 +  # normalise: 0.2 q/turn = 1.0
            0.20 * (st["turn_count"] / total_turns) +
            0.15 * min(st["word_count"] / max(st["turn_count"], 1) / 30, 1.0) +
            0.15 * 0.5  # initiative: no reliable signal in v1, use 50% default
        )
        ses = _clamp(ses_raw * 10)

        # Parent Engagement Score (0–10)
        pa = stats[Speaker.PARENT]
        pes_raw = (
            0.35 * min((pa["word_count"] / total_words) * 3, 1.0) +
            0.30 * min(pa["questions"] / max(total_questions, 1) * 3, 1.0) +
            0.20 * (0.8 if pa["objections"] > 0 else 0.3) +
            0.15 * min(pa["decisions"], 3) / 3
        )
        pes = _clamp(pes_raw * 10) if parent_present else 0.0

        # Stakeholder Coverage Score (0–10)
        scs = 5.0 if student_present else 0.0
        if parent_present and pa["turn_count"] > 2:
            scs += 3.0
        if ses > 6 and pes > 5:
            scs += 2.0
        scs = _clamp(scs)

        # Participation Quality (composite)
        pqs = _clamp(ses * 0.4 + pes * 0.3 + scs * 0.3)

        # ── 4. Flags ─────────────────────────────────────────────────────────
        flags: list[str] = []
        if not parent_present:
            flags.append("parent_absent")
        elif pes < 3:
            flags.append("parent_silent")
        if ses < 4:
            flags.append("student_passive")
        if ctr > 0.75:
            flags.append("counsellor_dominated")
        if pa["objections"] > 0 and pa["objections"] > (pa.get("decisions", 0) * 2):
            flags.append("unresolved_parent_objections")

        # ── 5. Participant profiles ───────────────────────────────────────────
        role_map = {
            Speaker.COUNSELLOR: "Counsellor",
            Speaker.STUDENT:    "Student",
            Speaker.PARENT:     "Parent",
            Speaker.UNKNOWN:    "Unknown",
        }
        participants = []
        for spk in [Speaker.COUNSELLOR, Speaker.STUDENT, Speaker.PARENT, Speaker.UNKNOWN]:
            s = stats[spk]
            if s["word_count"] == 0:
                continue
            att = _att_pct(spk) if spk != Speaker.COUNSELLOR else 100.0
            participants.append({
                "speaker_id":       role_map[spk],
                "role":             spk.value,
                "word_count":       s["word_count"],
                "word_share_pct":   round(s["word_count"] / total_words * 100, 1),
                "turn_count":       s["turn_count"],
                "questions_asked":  s["questions"],
                "objections_raised": s["objections"],
                "attendance_pct":   att,
            })

        # ── 6. Narrative ──────────────────────────────────────────────────────
        narrative = self._generate_narrative(
            ses, pes, ctr, pa, st, parent_present, flags, quality
        )

        return {
            "participants": participants,
            "attendance": {
                "student_present":        student_present,
                "parent_present":         parent_present,
                "student_attendance_pct": student_att,
                "parent_attendance_pct":  parent_att,
                "meeting_attendance_quality": quality,
            },
            "student_engagement_score":    round(ses, 2),
            "parent_engagement_score":     round(pes, 2),
            "counsellor_talk_ratio":       round(ctr, 3),
            "stakeholder_coverage_score":  round(scs, 2),
            "participation_quality_score": round(pqs, 2),
            "audit_impact_narrative":      narrative,
            "flags":                       flags,
        }

    # ── Narrative generator ───────────────────────────────────────────────────

    def _generate_narrative(
        self, ses: float, pes: float, ctr: float, pa: dict, st: dict,
        parent_present: bool, flags: list[str], quality: str,
    ) -> str:
        try:
            return self._llm_narrative(ses, pes, ctr, pa, st, parent_present, flags)
        except Exception as exc:
            logger.warning(f"LLM narrative failed, using rule-based: {exc}")
            return self._rule_narrative(ses, pes, ctr, parent_present, flags, quality)

    def _llm_narrative(
        self, ses: float, pes: float, ctr: float, pa: dict, st: dict,
        parent_present: bool, flags: list[str],
    ) -> str:
        if not settings.openai_api_key:
            raise ValueError("No OpenAI key")

        from openai import OpenAI
        client = OpenAI(api_key=settings.openai_api_key)

        summary = (
            f"Student engagement score: {ses:.1f}/10 ({st['turn_count']} turns, {st['questions']} questions asked). "
            f"Parent present: {parent_present}. "
            + (f"Parent engagement score: {pes:.1f}/10 ({pa['turn_count']} turns, {pa['questions']} questions, "
               f"{pa['objections']} objections raised). " if parent_present else "")
            + f"Counsellor talk ratio: {ctr*100:.0f}%. "
            + (f"Flags: {', '.join(flags)}. " if flags else "")
        )

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert sales coaching analyst for a college counselling context. "
                        "Write exactly 2-3 sentences about participant engagement quality. "
                        "Be specific, actionable, and concise. No bullet points."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Participation data: {summary}\nWrite the audit impact summary.",
                },
            ],
            max_tokens=120,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()

    @staticmethod
    def _rule_narrative(
        ses: float, pes: float, ctr: float,
        parent_present: bool, flags: list[str], quality: str,
    ) -> str:
        parts: list[str] = []

        if ses >= 7:
            parts.append(f"Student was actively engaged (score {ses:.1f}/10).")
        elif ses >= 5:
            parts.append(f"Student showed moderate engagement (score {ses:.1f}/10) — more open-ended questions could help.")
        else:
            parts.append(f"Student was largely passive (score {ses:.1f}/10) — counsellor should draw them out more.")

        if not parent_present:
            parts.append("Parent was not present — a parent-inclusive follow-up session is strongly recommended.")
        elif pes >= 6:
            parts.append(f"Parent was meaningfully engaged (score {pes:.1f}/10).")
        elif pes >= 3:
            parts.append(f"Parent participation was limited (score {pes:.1f}/10) — ensure their concerns are explicitly addressed.")
        else:
            parts.append(f"Parent was largely silent (score {pes:.1f}/10) — direct questions to the parent are essential.")

        if ctr > 0.75:
            parts.append(f"Counsellor talk ratio ({ctr*100:.0f}%) was too high — aim for 45–60% to create a two-way dialogue.")
        elif ctr < 0.35:
            parts.append(f"Counsellor spoke only {ctr*100:.0f}% — more structured guidance is needed.")

        return " ".join(parts[:3])

    @staticmethod
    def _empty_result() -> dict:
        return {
            "participants": [],
            "attendance": {
                "student_present": False,
                "parent_present": False,
                "student_attendance_pct": 0.0,
                "parent_attendance_pct": 0.0,
                "meeting_attendance_quality": "unknown",
            },
            "student_engagement_score": 0.0,
            "parent_engagement_score": 0.0,
            "counsellor_talk_ratio": 0.0,
            "stakeholder_coverage_score": 0.0,
            "participation_quality_score": 0.0,
            "audit_impact_narrative": "No utterances available for participant analysis.",
            "flags": ["no_utterances"],
        }
