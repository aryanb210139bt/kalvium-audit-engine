"""
intelligence/analyzers.py
Step 6b — Deterministic rule-based analysis layer (Layer 1 of hybrid audit).
Covers: script compliance, talk ratio, intent detection, objection analysis.
"""
from __future__ import annotations
import re
import logging
from collections import defaultdict

from config.models import (
    Utterance, StructuredEvent, ConversationEvent, Speaker, Sentiment,
    IntentType, ScriptComplianceResult, TalkRatioResult,
    IntentResult, SentimentResult, ObjectionResult, ObjectionAnalysis,
)

logger = logging.getLogger(__name__)


# ── 1. Script Compliance Analyzer ─────────────────────────────────────────────

class ScriptComplianceAnalyzer:
    """
    Layer 1: Binary checks — was each required demo stage completed?
    Uses structured events so we don't need LLM for this layer.
    """

    REQUIRED_EVENTS = {
        "counsellor_intro":        ConversationEvent.COUNSELLOR_INTRO,
        "kalvium_explanation":     ConversationEvent.KALVIUM_EXPLANATION,
        "traditional_comparison":  ConversationEvent.TRADITIONAL_COLLEGE_COMPARISON,
        "career_outcomes":         ConversationEvent.CAREER_OUTCOMES,
        "repo_shown":              ConversationEvent.REPO_SHOWN,
        "ppt_presented":           ConversationEvent.PPT_PRESENTED,
        "placements_discussed":    ConversationEvent.PLACEMENTS_DISCUSSION,
        "fee_structure_explained": ConversationEvent.FEE_STRUCTURE,
        "roi_explained":           ConversationEvent.ROI_EXPLANATION,
        "discovery_questions_asked": ConversationEvent.DISCOVERY_QUESTIONS,
        "closing_attempted":       ConversationEvent.CLOSING_ATTEMPT,
    }

    def analyze(self, events: list[StructuredEvent]) -> ScriptComplianceResult:
        counsellor_events = {
            e.event for e in events if e.speaker == Speaker.COUNSELLOR
        }

        result = ScriptComplianceResult()
        for field, required_event in self.REQUIRED_EVENTS.items():
            setattr(result, field, required_event in counsellor_events)

        logger.info(f"Script compliance: {result.completion_rate}% complete")
        return result


# ── 2. Talk Ratio Analyzer ─────────────────────────────────────────────────────

class TalkRatioAnalyzer:
    """
    Measures conversation balance: who talks how much, questions asked, silence.
    """

    QUESTION_PATTERNS = [
        r"\?\s*$", r"\bwhat\b", r"\bwhy\b", r"\bhow\b",
        r"\bwhen\b", r"\bwhere\b", r"\bwhich\b", r"\btell me\b",
    ]

    def analyze(self, utterances: list[Utterance]) -> TalkRatioResult:
        durations: dict[Speaker, float] = defaultdict(float)
        questions_by_speaker: dict[Speaker, int] = defaultdict(int)
        prev_end = 0.0
        dead_air = 0.0
        response_latencies: list[float] = []
        interruptions = 0
        prev_speaker = None

        for utt in sorted(utterances, key=lambda u: u.start_time):
            duration = utt.end_time - utt.start_time
            durations[utt.speaker] += duration

            # Dead air (silence between speakers)
            gap = utt.start_time - prev_end
            if 1.5 < gap < 10.0:      # ignore tiny pauses and long breaks
                dead_air += gap
            elif 0 < gap < 0.3 and prev_speaker != utt.speaker:
                interruptions += 1     # very quick turn = likely interruption
            elif gap > 0 and prev_speaker != utt.speaker:
                response_latencies.append(gap)

            # Question detection
            text = utt.english_text or utt.native_text
            is_question = any(re.search(p, text, re.IGNORECASE) for p in self.QUESTION_PATTERNS)
            if is_question:
                questions_by_speaker[utt.speaker] += 1

            prev_end = utt.end_time
            prev_speaker = utt.speaker

        total_duration = sum(durations.values()) or 1.0
        c_pct = round(durations[Speaker.COUNSELLOR] / total_duration * 100, 1)
        s_pct = round(durations[Speaker.STUDENT] / total_duration * 100, 1)
        p_pct = round(durations[Speaker.PARENT] / total_duration * 100, 1)

        total_questions = sum(questions_by_speaker.values())
        counsellor_q = questions_by_speaker[Speaker.COUNSELLOR]
        avg_latency = (
            sum(response_latencies) / len(response_latencies)
            if response_latencies else 0.0
        )

        result = TalkRatioResult(
            counsellor_pct=c_pct,
            student_pct=s_pct,
            parent_pct=p_pct,
            total_questions_asked=total_questions,
            counsellor_questions=counsellor_q,
            interruption_count=interruptions,
            avg_response_latency_sec=round(avg_latency, 2),
            dead_air_sec=round(dead_air, 1),
            is_balanced=c_pct < 70,
        )
        logger.info(
            f"Talk ratio — Counsellor: {c_pct}%, Student: {s_pct}%, Parent: {p_pct}% | "
            f"Questions: {total_questions} | Interruptions: {interruptions}"
        )
        return result


# ── 3. Intent Detector ─────────────────────────────────────────────────────────

class IntentDetector:
    """
    Detects high-level intent of student and parent from utterance content.
    Layer 1 uses keyword rules; Layer 2 (LLM) refines this.
    """

    # Patterns per intent type
    INTENT_SIGNALS: dict[IntentType, list[str]] = {
        IntentType.HIGH_INTEREST:      [r"how.*apply", r"when.*start", r"interested", r"sounds.*great", r"want to join"],
        IntentType.FEE_CONCERN:        [r"too expensive", r"can't afford", r"fee.*high", r"payment.*problem", r"financial"],
        IntentType.PLACEMENT_CONCERN:  [r"job.*guarantee", r"placement.*sure", r"what if.*no job", r"employment"],
        IntentType.CREDIBILITY_CONCERN:[r"aicte", r"recognised", r"accredited", r"valid.*degree", r"approved"],
        IntentType.PARENT_RESISTANCE:  [r"traditional.*safer", r"known.*college", r"risk.*new", r"not convinced"],
        IntentType.SCHOLARSHIP_INTEREST:[r"scholarship", r"financial.*aid", r"discount", r"fee.*waiver"],
        IntentType.STUDENT_EXCITED:    [r"excited", r"can't wait", r"love.*idea", r"definitely", r"sign.*up"],
        IntentType.STUDENT_PASSIVE:    [r"okay", r"hmm", r"i.*see", r"not sure", r"let me.*think"],
        IntentType.PARENT_SKEPTICAL:   [r"prove it", r"how do we know", r"guarantee", r"what.*proof", r"show.*evidence"],
        IntentType.CONFUSED:           [r"don't understand", r"what.*mean", r"confused", r"not clear", r"can.*explain"],
    }

    def analyze(
        self, utterances: list[Utterance], events: list[StructuredEvent]
    ) -> IntentResult:
        # Split utterances by speaker role
        student_texts = [u.english_text for u in utterances if u.speaker == Speaker.STUDENT]
        parent_texts  = [u.english_text for u in utterances if u.speaker == Speaker.PARENT]

        student_intent = self._classify_intent(student_texts)
        parent_intent  = self._classify_intent(parent_texts)

        # Key concerns (unique signals mentioned)
        key_concerns: list[str] = []
        for utt in utterances:
            if utt.speaker in (Speaker.STUDENT, Speaker.PARENT):
                text = utt.english_text.lower()
                if any(re.search(p, text) for p in self.INTENT_SIGNALS[IntentType.FEE_CONCERN]):
                    key_concerns.append("Fee affordability")
                if any(re.search(p, text) for p in self.INTENT_SIGNALS[IntentType.PLACEMENT_CONCERN]):
                    key_concerns.append("Placement guarantee")
                if any(re.search(p, text) for p in self.INTENT_SIGNALS[IntentType.CREDIBILITY_CONCERN]):
                    key_concerns.append("AICTE/accreditation")
        key_concerns = list(dict.fromkeys(key_concerns))   # deduplicate

        # Alignment risk
        risk = self._alignment_risk(student_intent, parent_intent)

        # Admission probability (simple heuristic, LLM refines)
        probability = self._estimate_probability(student_intent, parent_intent, events)

        # Overall intent = weighted combination
        overall = (
            student_intent
            if student_intent in (IntentType.HIGH_INTEREST, IntentType.STUDENT_EXCITED)
            else parent_intent
            if parent_intent == IntentType.PARENT_RESISTANCE
            else IntentType.NEUTRAL
        )

        result = IntentResult(
            student_intent=student_intent,
            parent_intent=parent_intent,
            overall_intent=overall,
            alignment_risk=risk,
            key_concerns=key_concerns,
            admission_probability=probability,
        )
        logger.info(
            f"Intent — Student: {student_intent.value}, Parent: {parent_intent.value}, "
            f"Risk: {risk}, P(admit): {probability:.0%}"
        )
        return result

    def _classify_intent(self, texts: list[str]) -> IntentType:
        if not texts:
            return IntentType.NEUTRAL
        combined = " ".join(texts).lower()
        scores: dict[IntentType, int] = defaultdict(int)
        for intent, patterns in self.INTENT_SIGNALS.items():
            scores[intent] = sum(1 for p in patterns if re.search(p, combined))
        best = max(scores, key=scores.__getitem__)
        return best if scores[best] > 0 else IntentType.NEUTRAL

    @staticmethod
    def _alignment_risk(student: IntentType, parent: IntentType) -> str:
        high_student = student in (IntentType.HIGH_INTEREST, IntentType.STUDENT_EXCITED)
        resistant_parent = parent in (IntentType.PARENT_RESISTANCE, IntentType.FEE_CONCERN, IntentType.PARENT_SKEPTICAL)
        if high_student and resistant_parent:
            return "high"
        elif resistant_parent or parent in (IntentType.CONFUSED,):
            return "medium"
        return "low"

    @staticmethod
    def _estimate_probability(
        student: IntentType, parent: IntentType, events: list[StructuredEvent]
    ) -> float:
        score = 0.5  # base
        if student == IntentType.HIGH_INTEREST:      score += 0.15
        if student == IntentType.STUDENT_EXCITED:    score += 0.10
        if student == IntentType.STUDENT_PASSIVE:    score -= 0.10
        if parent == IntentType.HIGH_INTEREST:       score += 0.15
        if parent == IntentType.PARENT_RESISTANCE:   score -= 0.20
        if parent == IntentType.FEE_CONCERN:         score -= 0.10
        if parent == IntentType.PARENT_SKEPTICAL:    score -= 0.10

        # Closing attempt detected → positive signal
        closing_events = {e.event for e in events}
        if ConversationEvent.CLOSING_ATTEMPT in closing_events:
            score += 0.05
        if ConversationEvent.FEE_STRUCTURE in closing_events:
            score += 0.05   # got this far in conversation

        return round(max(0.0, min(1.0, score)), 2)


# ── 4. Sentiment Analyzer ──────────────────────────────────────────────────────

class SentimentAnalyzer:
    """
    Multi-level sentiment analysis.
    Uses XLM-RoBERTa when available, otherwise falls back to rule-based.
    """

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._use_ml = self._try_load_model()

    def _try_load_model(self) -> bool:
        try:
            from transformers import pipeline as hf_pipeline
            self._hf_pipeline = hf_pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
                top_k=None,
            )
            logger.info("XLM-RoBERTa sentiment model loaded")
            return True
        except Exception as e:
            logger.warning(f"Could not load XLM-RoBERTa ({e}) — using rule-based sentiment")
            return False

    def analyze(self, utterances: list[Utterance]) -> SentimentResult:
        by_speaker: dict[Speaker, list[Sentiment]] = defaultdict(list)
        timeline: list[dict] = []

        for utt in utterances:
            text = utt.english_text or utt.native_text
            if not text.strip():
                continue
            sent = self._classify(text)
            by_speaker[utt.speaker].append(sent)
            timeline.append({
                "timestamp": utt.start_time,
                "speaker": utt.speaker.value,
                "sentiment": sent.value,
                "text_preview": text[:60],
            })

        def majority(sentiments: list[Sentiment]) -> Sentiment:
            if not sentiments:
                return Sentiment.NEUTRAL
            from collections import Counter
            counts = Counter(sentiments)
            return counts.most_common(1)[0][0]

        all_sentiments = [s for sentiments in by_speaker.values() for s in sentiments]
        result = SentimentResult(
            overall=majority(all_sentiments),
            counsellor_sentiment=majority(by_speaker[Speaker.COUNSELLOR]),
            student_sentiment=majority(by_speaker[Speaker.STUDENT]),
            parent_sentiment=majority(by_speaker[Speaker.PARENT]),
            sentiment_timeline=timeline,
        )
        logger.info(
            f"Sentiment — Overall: {result.overall.value}, "
            f"Counsellor: {result.counsellor_sentiment.value}, "
            f"Student: {result.student_sentiment.value}, "
            f"Parent: {result.parent_sentiment.value}"
        )
        return result

    def _classify(self, text: str) -> Sentiment:
        if self._use_ml:
            try:
                results = self._hf_pipeline(text[:512])
                if results and results[0]:
                    scores = {r["label"].lower(): r["score"] for r in results[0]}
                    if scores.get("positive", 0) > 0.5:
                        return Sentiment.POSITIVE
                    elif scores.get("negative", 0) > 0.5:
                        return Sentiment.NEGATIVE
                    return Sentiment.NEUTRAL
            except Exception:
                pass
        return self._rule_based(text)

    @staticmethod
    def _rule_based(text: str) -> Sentiment:
        from intelligence.event_detector import detect_sentiment
        return detect_sentiment(text)


# ── 5. Objection Analyzer ──────────────────────────────────────────────────────

class ObjectionAnalyzer:
    """
    Detects objections raised and evaluates whether they were handled.
    """

    OBJECTION_CATEGORIES = {
        "Fee too high": [r"too expensive", r"can't afford", r"fee.*high", r"costly"],
        "Placement uncertainty": [r"no.*job", r"placement.*guarantee", r"what if.*unemployed"],
        "AICTE/recognition": [r"aicte", r"recognised", r"valid degree", r"accredited"],
        "Risk of new college": [r"new.*college", r"risky", r"not.*established"],
        "Traditional college safer": [r"traditional.*safer", r"known.*college", r"iit\b", r"nit\b"],
        "Parent disapproval": [r"parent.*no", r"mom.*not", r"dad.*not", r"family.*not"],
        "ROI doubt": [r"worth.*money", r"not worth", r"return.*investment"],
    }

    RESOLUTION_SIGNALS = [
        r"actually\b", r"let me explain\b", r"our data shows\b",
        r"100%\b.*placed", r"fully approved\b", r"aicte approved\b",
        r"scholarship\b.*available", r"many student\b", r"proven\b",
        r"testimonial\b", r"let me share\b", r"data.*show\b",
        r"placed.*lakh\b", r"i understand.*concern\b",
    ]

    def analyze(
        self, utterances: list[Utterance], events: list[StructuredEvent]
    ) -> ObjectionAnalysis:
        objection_utterances = [
            u for u in utterances
            if u.speaker in (Speaker.STUDENT, Speaker.PARENT)
            and any(
                any(re.search(p, u.english_text, re.IGNORECASE) for p in patterns)
                for patterns in self.OBJECTION_CATEGORIES.values()
            )
        ]

        results: list[ObjectionResult] = []
        for obj_utt in objection_utterances:
            category = self._categorize(obj_utt.english_text)
            # Look for resolution in the next 3 counsellor utterances
            obj_idx = next(
                (i for i, u in enumerate(utterances) if u.utterance_id == obj_utt.utterance_id),
                -1,
            )
            resolution_quality = self._check_resolution(utterances, obj_idx)

            from intelligence.event_detector import seconds_to_timestamp
            results.append(ObjectionResult(
                objection_text=obj_utt.english_text[:200],
                category=category,
                timestamp=seconds_to_timestamp(obj_utt.start_time),
                was_addressed=resolution_quality != "missed",
                resolution_quality=resolution_quality,
            ))

        resolved     = sum(1 for r in results if r.resolution_quality == "well")
        partial      = sum(1 for r in results if r.resolution_quality == "partial")
        missed       = sum(1 for r in results if r.resolution_quality == "missed")

        analysis = ObjectionAnalysis(
            total_objections=len(results),
            resolved=resolved,
            partially_resolved=partial,
            missed=missed,
            objections=results,
        )
        logger.info(
            f"Objections — Total: {len(results)}, Resolved: {resolved}, "
            f"Partial: {partial}, Missed: {missed}"
        )
        return analysis

    def _categorize(self, text: str) -> str:
        text_lower = text.lower()
        for category, patterns in self.OBJECTION_CATEGORIES.items():
            if any(re.search(p, text_lower) for p in patterns):
                return category
        return "General concern"

    def _check_resolution(self, utterances: list[Utterance], obj_idx: int) -> str:
        """Check whether counsellor addressed the objection in the next few turns."""
        if obj_idx < 0:
            return "missed"
        followup_counsellor = [
            u for u in utterances[obj_idx + 1: obj_idx + 6]
            if u.speaker == Speaker.COUNSELLOR
        ]
        if not followup_counsellor:
            return "missed"
        combined = " ".join(u.english_text for u in followup_counsellor).lower()
        signal_count = sum(1 for p in self.RESOLUTION_SIGNALS if re.search(p, combined))
        if signal_count >= 2:
            return "well"
        elif signal_count == 1:
            return "partial"
        return "missed"
