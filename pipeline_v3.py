"""
pipeline_v3.py
Production pipeline with:
  - 10-category evaluation framework (KNET-first, Indian context)
  - Advanced multilingual sentiment (XLM-RoBERTa + GPT)
  - Parallel STT (8× workers)
  - Parallel GPT-4o audit (8× workers)
  - Progress events for real-time dashboard
"""
from __future__ import annotations
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from config.models import Speaker, Utterance
from config.settings import get_settings
from progress_tracker import ProgressTracker

from audio.processor import AudioProcessor
from transcription.stt import SpeechToText
from diarization.diarizer import Diarizer
from intelligence.event_detector import ConversationEventDetector
from intelligence.analyzers import (
    ScriptComplianceAnalyzer, TalkRatioAnalyzer,
    IntentDetector, ObjectionAnalyzer,
)
from intelligence.sentiment_v2 import AdvancedSentimentAnalyzer
from audit.explainable_auditor import ExplainableAuditor
from audit.evaluation_framework import EVALUATION_CATEGORIES
from audit.ppt_coverage import check_ppt_coverage, ppt_coverage_to_dict
from scoring.scorer_v2 import ScoringEngineV2, CoachingGeneratorV2

logger = logging.getLogger(__name__)
settings = get_settings()

STT_WORKERS   = 8
AUDIT_WORKERS = 8


class DemoAuditPipelineV3:

    def __init__(self, progress: ProgressTracker | None = None):
        self.p          = progress
        self.processor  = AudioProcessor()
        self.stt        = SpeechToText()
        self.diarizer   = Diarizer()
        self.event_det  = ConversationEventDetector()
        self.compliance = ScriptComplianceAnalyzer()
        self.talk_ratio = TalkRatioAnalyzer()
        self.intent     = IntentDetector()
        self.sentiment  = AdvancedSentimentAnalyzer(use_xlm=True, use_gpt=True)
        self.objections = ObjectionAnalyzer()
        self.llm        = ExplainableAuditor(progress=progress)
        self.scorer     = ScoringEngineV2()
        self.coach      = CoachingGeneratorV2()

    def run(self, recording_path: str | Path, session_id: str | None = None) -> dict:
        """
        Returns a dict (DemoAuditReport.model_dump() + advanced_sentiment + category_coaching)
        so that reports can be generated directly from it.

        session_id: pass the caller's own audit/job id (api/main.py's
        _sessions/job_queue key) so transcription/sarvam_job_store persists
        the Sarvam Batch STT job under the SAME id used everywhere else —
        this is what lets Resume/Retry (POST /api/v1/queue/{id}/retry) pick
        up the already-submitted Sarvam job instead of resubmitting it, and
        what lets Cancel cooperatively stop the poll loop for this session.
        Defaults to a fresh uuid4 for callers that don't have (or don't
        care about) an external id, e.g. ad-hoc/test invocations.
        """
        recording_path = Path(recording_path)
        session_id     = session_id or str(uuid.uuid4())
        started_at     = datetime.utcnow()

        logger.info(f"Pipeline V3 start — {recording_path.name} (session {session_id})")
        if self.p: self.p.log(f"Starting V3 audit: {recording_path.name}")

        # ── Step 1: Audio → WAV (no chunking — Sarvam Batch STT takes the
        # whole file in one job; see transcription/sarvam_batch.py) ──────────
        if self.p: self.p.step_start(1, f"Converting {recording_path.name}")
        wav_path = self.processor.to_wav(recording_path)
        duration = self.processor.get_duration(wav_path)
        if self.p:
            self.p.step_done(1, f"{duration:.0f}s audio")

        # ── Step 2: decide single Batch STT job vs >2h split ──────────────────
        from transcription.sarvam_batch import BATCH_MAX_SECONDS
        if duration <= BATCH_MAX_SECONDS:
            if self.p: self.p.step_start(2, "≤2h recording — single Sarvam Batch STT job")
            segments = [(wav_path, 0.0)]
            if self.p: self.p.step_done(2, "1 job")
        else:
            if self.p: self.p.step_start(
                2, f"{duration/3600:.1f}h recording exceeds the Batch API's 2h/file limit "
                   f"— splitting into valid-sized segments")
            raw_segments = self.processor.split_for_batch(wav_path, BATCH_MAX_SECONDS)
            segments = [(Path(c.file_path), c.start_time) for c in raw_segments]
            if self.p: self.p.step_done(2, f"{len(segments)} segment(s)")

        # ── Step 3: Sarvam Batch STT — initialise → upload → start → poll →
        # download, per segment (almost always just one) ─────────────────────
        if self.p: self.p.step_start(3, "Sarvam Batch STT → Job submitted")
        utterances = self._batch_stt(session_id, segments)
        if self.p: self.p.step_done(3, f"{len(utterances)} utterances — Transcript ready")

        # ── Step 4: Speaker labels are already assigned per-utterance in
        # _batch_stt (from Sarvam's diarized_transcript, mapped to
        # Counsellor/Student/Parent via the existing talk-time heuristic —
        # or the existing acoustic diarizer as fallback if Sarvam's beta
        # diarization didn't return segments). This step just reports it. ──
        if self.p: self.p.step_start(4, "Speaker labels from diarization")
        speakers      = sorted({u.speaker.value for u in utterances})
        language_det  = utterances[0].language_detected if utterances else "en"
        if self.p: self.p.step_done(4, f"Speakers: {', '.join(speakers)}")

        # ── Step 4b: Strip setup/audio-check utterances ───────────────────────
        utterances = self._strip_setup_utterances(utterances)
        if self.p: self.p.log(f"Transcript after setup-strip: {len(utterances)} utterances")

        # ── Step 4c: Normalise STT mis-transcriptions ─────────────────────────
        from utils.word_normalizer import normalize_utterances
        utterances = normalize_utterances(utterances)

        # ── Step 5: Rule-based analysis + advanced sentiment ──────────────────
        if self.p: self.p.step_start(5, f"Rule analysis + XLM-RoBERTa sentiment")
        events     = self.event_det.detect(utterances)
        compliance = self.compliance.analyze(events)
        talk       = self.talk_ratio.analyze(utterances)
        intent     = self.intent.analyze(utterances, events)
        objection  = self.objections.analyze(utterances, events)
        adv_sent   = self.sentiment.analyze(utterances)
        if self.p:
            self.p.step_done(
                5,
                f"{len(events)} events | {objection.total_objections} objections | "
                f"Sentiment: {adv_sent.overall_sentiment} | Momentum: {adv_sent.emotional_momentum}"
            )

        # ── Step 5b: Deck Coverage (semantic) + legacy PPT fallback ─────────────
        deck_coverage_report = None
        try:
            from deck import db as deck_db
            from deck.evaluation.semantic_aligner import (
                align_transcript_to_deck, coverage_report_to_dict
            )
            active_deck = deck_db.get_active_deck()
            if active_deck:
                if self.p: self.p.log(
                    f"Running semantic deck alignment: '{active_deck['deck_name']}' "
                    f"({active_deck['total_criteria']} criteria)…"
                )
                deck_coverage_report = align_transcript_to_deck(
                    session_id, utterances, active_deck["deck_id"]
                )
                cov = deck_coverage_report
                if self.p:
                    self.p.log(
                        f"Deck coverage: {cov.must_covered}/{cov.must_total} MUST | "
                        f"{cov.should_covered}/{cov.should_total} SHOULD | "
                        f"Score: {cov.coverage_score:.0f}/100"
                    )
                    self.p.emit(
                        "deck_coverage",
                        deck_id=active_deck["deck_id"],
                        deck_name=active_deck["deck_name"],
                        coverage_score=cov.coverage_score,
                        must_covered=cov.must_covered,
                        must_total=cov.must_total,
                        should_covered=cov.should_covered,
                        should_total=cov.should_total,
                        missed_must=cov.missed_must[:5],
                        missed_should=cov.missed_should[:5],
                        results=[
                            {"criterion_id": r.criterion_id, "label": r.label,
                             "importance": r.importance, "slide_number": r.slide_number,
                             "covered": r.covered, "similarity": r.best_similarity}
                            for r in cov.results
                        ],
                    )
            else:
                # Fallback to legacy keyword-based PPT check
                if self.p: self.p.log("No active deck — using legacy PPT keyword check…")
                ppt_report = check_ppt_coverage(utterances)
                if self.p:
                    self.p.emit(
                        "ppt_coverage",
                        coverage_score=ppt_report.coverage_score,
                        must_covered=ppt_report.must_covered,
                        must_total=ppt_report.must_total,
                        should_covered=ppt_report.should_covered,
                        should_total=ppt_report.should_total,
                        missed_must=ppt_report.missed_must,
                        missed_should=ppt_report.missed_should[:4],
                        sections=[
                            {"id": s.section_id, "label": s.label,
                             "priority": s.priority, "covered": s.covered}
                            for s in ppt_report.sections
                        ],
                    )
        except Exception as _deck_err:
            logger.warning(f"Deck coverage step failed (non-fatal): {_deck_err}")

        # ── Step 6: Parallel 10-category GPT-4o audit ────────────────────────
        if self.p: self.p.step_start(6, f"10 GPT-4o evaluations ({AUDIT_WORKERS}× parallel)")
        llm_results = self.llm.evaluate(utterances, events, compliance, talk, objection)
        if self.p: self.p.step_done(6, f"All {len(llm_results)} categories evaluated")

        # ── Step 7: Score + coaching + composite ─────────────────────────────
        if self.p: self.p.step_start(7, "Computing weighted score + coaching")
        score = self.scorer.compute_score(compliance, talk, intent, objection, llm_results)
        hi, st, im, cat_coaching = self.coach.generate(score, compliance, talk, objection, llm_results)

        # Composite score (behavioral + deck coverage)
        from deck.evaluation.composite_scorer import compute_composite
        deck_cov_score = deck_coverage_report.coverage_score if deck_coverage_report else None
        try:
            from deck import db as deck_db
            active_deck_meta = deck_db.get_active_deck()
            deck_weights = active_deck_meta.get("scoring_weights") if active_deck_meta else None
            active_deck_id = active_deck_meta["deck_id"] if active_deck_meta else None
        except Exception:
            deck_weights = None
            active_deck_id = None

        composite = compute_composite(
            behavioral_score    = score.overall,
            deck_coverage_score = deck_cov_score,
            deck_id             = active_deck_id,
            scoring_weights     = deck_weights,
        )
        elapsed = (datetime.utcnow() - started_at).total_seconds()

        if self.p:
            self.p.step_done(
                7,
                f"Behavioral: {score.overall}/100 | "
                f"Deck: {deck_cov_score:.0f}/100 | " if deck_cov_score else "",
            )
            self.p.step_done(7, f"Final: {composite.overall}/100 ({composite.grade}) in {elapsed:.0f}s")
            explainable_payload = [
                {
                    "category": r.category,
                    "category_id": r.category_id,
                    "score": r.score,
                    "why_score": r.why_score,
                    "reason_for_zero": r.reason_for_zero,
                    "evidence_found": r.evidence_found[:3],
                    "missing_behaviors": r.missing_behaviors[:3],
                    "transcript_evidence": [
                        {"timestamp": te.timestamp, "speaker": te.speaker,
                         "quote": te.quote[:120], "relevance": te.relevance}
                        for te in r.transcript_evidence[:2]
                    ],
                    "expected_behaviors": r.expected_behaviors[:3],
                    "coaching_feedback": r.coaching_feedback,
                    "score_confidence": r.score_confidence,
                    "evaluation_basis": r.evaluation_basis,
                }
                for r in llm_results
            ]
            self.p.score_ready(
                overall   = composite.overall,
                grade     = composite.grade,
                scores    = {cat_id: cs.raw_score for cat_id, cs in score.category_scores.items()},
                highlights= hi,
                strengths = st,
                improvements = im,
                explainable  = explainable_payload,
                composite = {
                    "overall":              composite.overall,
                    "behavioral_score":     composite.behavioral_score,
                    "behavioral_component": composite.behavioral_component,
                    "deck_coverage_score":  composite.deck_coverage_score,
                    "deck_component":       composite.deck_component,
                    "deck_id":              composite.deck_id,
                    "weights":              composite.weights_used,
                },
            )
            self.p.finish()

        logger.info(f"Pipeline V3 done in {elapsed:.0f}s — {score.overall}/100 ({score.grade})")

        # ── Step 7b: Participant Intelligence ─────────────────────────────────
        pi_dict = None
        try:
            from intelligence.participant_analyzer import ParticipantAnalyzer
            if self.p: self.p.log("Running participant intelligence analysis…")
            pi_dict = ParticipantAnalyzer().analyze(utterances, duration)
            if self.p:
                att = pi_dict.get("attendance", {})
                self.p.log(
                    f"Participant: SES={pi_dict['student_engagement_score']:.1f} | "
                    f"PES={pi_dict['parent_engagement_score']:.1f} | "
                    f"CTR={pi_dict['counsellor_talk_ratio']*100:.0f}% | "
                    f"Parent={'✓' if att.get('parent_present') else '✗'}"
                )
        except Exception as _pi_err:
            logger.warning(f"Participant analysis (non-fatal): {_pi_err}")

        # Build core report (compatible with DemoAuditReport structure)
        from dataclasses import asdict
        import dataclasses

        def _arc_to_dict(arc):
            if arc is None: return None
            if dataclasses.is_dataclass(arc): return dataclasses.asdict(arc)
            return arc

        adv_sent_dict = {
            "overall_sentiment": adv_sent.overall_sentiment,
            "student_excited": adv_sent.student_excited,
            "parent_convinced": adv_sent.parent_convinced,
            "emotional_momentum": adv_sent.emotional_momentum,
            "engagement_score": adv_sent.engagement_score,
            "emotional_alignment_score": adv_sent.emotional_alignment_score,
            "gpt_sentiment_narrative": adv_sent.gpt_sentiment_narrative,
            "admission_likelihood_from_sentiment": adv_sent.admission_likelihood_from_sentiment,
            "counsellor_arc": _arc_to_dict(adv_sent.counsellor_arc),
            "student_arc": _arc_to_dict(adv_sent.student_arc),
            "parent_arc": _arc_to_dict(adv_sent.parent_arc),
            "turning_points": [dataclasses.asdict(tp) for tp in adv_sent.turning_points],
            "sentiment_timeline": adv_sent.sentiment_timeline,
        }

        report_dict = {
            "session_id": session_id,
            "created_at": started_at.isoformat(),
            "recording_file": str(recording_path),
            "duration_seconds": duration,
            "language_detected": language_det,
            "speakers_detected": speakers,
            "utterances": [u.model_dump() for u in utterances],
            "structured_events": [e.model_dump() for e in events],
            "script_compliance": compliance.model_dump(),
            "talk_ratio": talk.model_dump(),
            "intent": intent.model_dump(),
            "objections": objection.model_dump(),
            "llm_evaluations": [r.model_dump() for r in llm_results],   # ExplainableEvaluationResult dicts
            "score": {
                "overall": score.overall,
                "grade": score.grade,
                "admission_probability": score.admission_probability,
                "category_scores": {
                    cat_id: cs.model_dump()
                    for cat_id, cs in score.category_scores.items()
                },
                "strongest_categories": score.strongest_categories,
                "weakest_categories": score.weakest_categories,
            },
            "coaching_highlights": hi,
            "top_strengths": st,
            "improvement_areas": im,
            "category_coaching": cat_coaching,
            "advanced_sentiment": adv_sent_dict,
            "ppt_coverage": ppt_coverage_to_dict(ppt_report) if 'ppt_report' in dir() else {},
            # ── Deck-aware evaluation results ──────────────────────────────────
            "deck_coverage": (
                {
                    "deck_id":        active_deck_id,
                    "coverage_score": deck_coverage_report.coverage_score,
                    "must_covered":   deck_coverage_report.must_covered,
                    "must_total":     deck_coverage_report.must_total,
                    "should_covered": deck_coverage_report.should_covered,
                    "should_total":   deck_coverage_report.should_total,
                    "missed_must":    deck_coverage_report.missed_must,
                    "missed_should":  deck_coverage_report.missed_should,
                    "results": [
                        {"criterion_id": r.criterion_id, "label": r.label,
                         "importance": r.importance, "slide_number": r.slide_number,
                         "covered": r.covered, "similarity": r.best_similarity,
                         "mode": r.detection_mode}
                        for r in deck_coverage_report.results
                    ],
                }
                if deck_coverage_report else None
            ),
            "composite_score": {
                "overall":              composite.overall,
                "grade":                composite.grade,
                "behavioral_score":     composite.behavioral_score,
                "behavioral_component": composite.behavioral_component,
                "deck_coverage_score":  composite.deck_coverage_score,
                "deck_component":       composite.deck_component,
                "deck_id":              composite.deck_id,
                "weights":              composite.weights_used,
            },
            "participant_intelligence": pi_dict,
        }

        return report_dict

    # ── Transcript input (skip STT) ───────────────────────────────────────────

    def run_from_transcript(self, utterance_dicts: list[dict],
                            label: str = "") -> dict:
        """
        Run the full evaluation pipeline from pre-parsed utterances.
        Skips Steps 1-4 (audio processing, chunking, STT, diarization).
        Picks up at Step 5 (event detection) and runs everything normally.
        """
        session_id = str(uuid.uuid4())
        started_at = datetime.utcnow()

        logger.info(f"Pipeline V3 (transcript mode) — {len(utterance_dicts)} utterances — {label}")
        if self.p: self.p.log(f"Transcript mode: {len(utterance_dicts)} utterances")

        # Convert dicts → Utterance objects
        utterances = []
        for i, d in enumerate(utterance_dicts):
            spk_val = d.get("speaker", "counsellor").lower()
            spk = Speaker.COUNSELLOR if "counsell" in spk_val or "agent" in spk_val else Speaker.STUDENT
            utterances.append(Utterance(
                utterance_id     = d.get("utterance_id", f"utt_{i:04d}"),
                speaker          = spk,
                native_text      = d.get("native_text", d.get("english_text", "")),
                english_text     = d.get("english_text", d.get("native_text", "")),
                start_time       = float(d.get("start_time", 0)),
                end_time         = float(d.get("end_time", 0)),
                language_detected= d.get("language_detected", "en"),
                confidence       = float(d.get("confidence", 1.0)),
            ))

        # Emit fake step events so the dashboard pipeline shows progress
        if self.p:
            self.p.step_start(1, "Transcript mode — skipping audio conversion")
            self.p.step_done(1, "N/A (transcript input)")
            self.p.step_start(2, "Transcript mode — skipping chunking")
            self.p.step_done(2, "N/A (transcript input)")
            self.p.step_start(3, f"Transcript parsed — {len(utterances)} utterances")
            self.p.step_done(3, f"{len(utterances)} utterances loaded")
            self.p.step_start(4, "Speaker labels from transcript")
            self.p.step_done(4, "Counsellor / Student assigned")

        # Strip setup utterances
        utterances = self._strip_setup_utterances(utterances)
        if self.p: self.p.log(f"After setup-strip: {len(utterances)} utterances")

        # Normalise STT mis-transcriptions
        from utils.word_normalizer import normalize_utterances
        utterances = normalize_utterances(utterances)

        # ── Step 5 onwards — identical to run() ──────────────────────────────
        if self.p: self.p.step_start(5, "Rule analysis + sentiment")
        events     = self.event_det.detect(utterances)
        compliance = self.compliance.analyze(events)
        talk       = self.talk_ratio.analyze(utterances)
        intent     = self.intent.analyze(utterances, events)
        objection  = self.objections.analyze(utterances, events)
        adv_sent   = self.sentiment.analyze(utterances)
        if self.p: self.p.step_done(5, f"Events: {len(events)}")

        if self.p: self.p.step_start(6, "Deck coverage")
        deck_coverage_report = None
        try:
            from deck import db as deck_db
            active_deck = deck_db.get_active_deck()
            if active_deck:
                from deck.evaluation.semantic_aligner import (
                    align_transcript_to_deck, coverage_report_to_dict
                )
                deck_coverage_report = align_transcript_to_deck(
                    session_id, utterances, active_deck["deck_id"]
                )
                if self.p:
                    cov = deck_coverage_report
                    self.p.emit("deck_coverage", **{
                        "session_id":    session_id,
                        "deck_id":       active_deck["deck_id"],
                        "deck_name":     active_deck.get("deck_name", ""),
                        "coverage_score":cov.coverage_score,
                        "must_covered":  cov.must_covered,
                        "must_total":    cov.must_total,
                        "should_covered":cov.should_covered,
                        "should_total":  cov.should_total,
                        "missed_must":   cov.missed_must,
                        "missed_should": cov.missed_should,
                        "results": [
                            {"criterion_id": r.criterion_id, "label": r.label,
                             "importance": r.importance, "covered": r.covered,
                             "similarity": r.best_similarity}
                            for r in cov.results
                        ],
                    })
        except Exception as e:
            logger.warning(f"Deck coverage skipped: {e}")
        if self.p: self.p.step_done(6, "Done" if deck_coverage_report else "No active deck")

        if self.p: self.p.step_start(7, "GPT-4o audit")
        if self.p: self.p.step_start(6, "GPT-4o audit (9 categories)")
        llm_results = self.llm.evaluate(utterances, events, compliance, talk, objection)
        if self.p: self.p.step_done(6, f"All 9 categories evaluated")

        if self.p: self.p.step_start(7, "Scoring")
        score = self.scorer.compute_score(compliance, talk, intent, objection, llm_results)
        elapsed = (datetime.utcnow() - started_at).total_seconds()

        from deck.evaluation.composite_scorer import compute_composite
        deck_cov_score = deck_coverage_report.coverage_score if deck_coverage_report else None
        deck_id_used   = deck_coverage_report.deck_id if deck_coverage_report else None
        composite = compute_composite(
            behavioral_score    = score.overall,
            deck_coverage_score = deck_cov_score,
            deck_id             = deck_id_used,
        )

        # Coaching
        hi, st_list, im = [], [], []
        try:
            coaching = self.coaching.generate(score, events, talk, compliance)
            hi, st_list, im = coaching.highlights, coaching.strengths, coaching.improvements
        except Exception: pass

        # Explainable payload
        explainable_payload = [
            {"category_id": r.category_id, "category_name": r.category_name,
             "score": r.raw_score, "reasoning": r.reasoning,
             "keywords_found": r.keywords_found, "expected_behaviors": r.expected_behaviors[:3],
             "coaching_feedback": r.coaching_feedback, "score_confidence": r.score_confidence,
             "evaluation_basis": r.evaluation_basis}
            for r in llm_results
        ]

        if self.p:
            self.p.step_done(7, f"Final: {composite.overall}/100 ({composite.grade})")
            self.p.score_ready(
                overall      = composite.overall,
                grade        = composite.grade,
                scores       = {cat_id: cs.raw_score for cat_id, cs in score.category_scores.items()},
                highlights   = hi,
                strengths    = st_list,
                improvements = im,
                explainable  = explainable_payload,
                composite = {
                    "overall":              composite.overall,
                    "behavioral_score":     composite.behavioral_score,
                    "behavioral_component": composite.behavioral_component,
                    "deck_coverage_score":  composite.deck_coverage_score,
                    "deck_component":       composite.deck_component,
                    "deck_id":              composite.deck_id,
                    "weights":              composite.weights_used,
                },
            )
            self.p.finish()

        logger.info(f"Transcript pipeline done in {elapsed:.0f}s — "
                    f"{composite.overall}/100 ({composite.grade})")

        # Build report dict (same structure as run())
        return self._build_report_dict(
            session_id, utterances, score, composite,
            events, talk, compliance, adv_sent, objection,
            deck_coverage_report, elapsed, label
        )

    def _build_report_dict(self, session_id, utterances, score, composite,
                           events, talk, compliance, adv_sent, objection,
                           deck_coverage_report, elapsed, label="") -> dict:
        """Shared report-building logic used by both run() and run_from_transcript()."""
        from dataclasses import asdict
        import dataclasses

        def _arc_to_dict(arc):
            if arc is None: return None
            if dataclasses.is_dataclass(arc): return dataclasses.asdict(arc)
            return arc

        adv_sent_dict = {
            "overall_sentiment": adv_sent.overall_sentiment if adv_sent else "neutral",
            "counsellor_sentiment": adv_sent.counsellor_sentiment if adv_sent else "neutral",
            "student_sentiment": adv_sent.student_sentiment if adv_sent else "neutral",
            "sentiment_timeline": [_arc_to_dict(s) for s in (adv_sent.timeline if adv_sent else [])],
        }
        from deck.evaluation.semantic_aligner import coverage_report_to_dict
        deck_cov_score = deck_coverage_report.coverage_score if deck_coverage_report else None
        deck_id_used   = deck_coverage_report.deck_id if deck_coverage_report else None

        return {
            "session_id":    session_id,
            "label":         label,
            "score": {
                "overall":         composite.overall,
                "grade":           composite.grade,
                "category_scores": {k: v.raw_score for k, v in score.category_scores.items()},
            },
            "utterances":    [u.model_dump() for u in utterances],
            "events":        [_arc_to_dict(e) for e in events],
            "talk_ratio":    _arc_to_dict(talk),
            "compliance":    _arc_to_dict(compliance),
            "sentiment":     adv_sent_dict,
            "objections":    _arc_to_dict(objection),
            "deck_coverage": coverage_report_to_dict(deck_coverage_report) if deck_coverage_report else None,
            "composite_score": {
                "overall":              composite.overall,
                "behavioral_score":     composite.behavioral_score,
                "behavioral_component": composite.behavioral_component,
                "deck_coverage_score":  composite.deck_coverage_score,
                "deck_component":       composite.deck_component,
                "deck_id":              composite.deck_id,
                "weights":              composite.weights_used,
            },
            "duration_seconds": elapsed,
        }

    # ── Sarvam Batch STT (current default STT path) ───────────────────────────

    def _batch_stt(self, session_id: str, segments: list[tuple[Path, float]]) -> list[Utterance]:
        """
        Runs the async Batch STT lifecycle per segment — almost always a
        single segment covering the whole recording (the ≤2h default case:
        one Sarvam job, no chunk-level bookkeeping needed at all). More than
        one segment only for >2h recordings split by
        AudioProcessor.split_for_batch, where N is always derived from the
        actual duration — never a fixed number, and never assumed small.

        For N==1, processed directly (no pre-created row, no worker pool —
        unchanged from the original single-job design). For N>1: all N
        segment rows are pre-created as 'pending' up front (so total/progress
        are always known from real persisted rows, not inferred), then
        processed through a bounded ThreadPoolExecutor
        (SARVAM_BATCH_SEGMENT_CONCURRENCY workers — decoupled from N; N=5 or
        N=500 both respect the same cap). Segments complete in whatever
        order their Sarvam jobs finish, not sequentially — progress is
        always COUNT(status=X) over sarvam_job_store's real rows (see
        segment_progress()), never "highest index seen". One segment failing
        does not stop the others; failures are collected and raised once
        every submitted segment has been attempted.
        """
        from transcription import sarvam_batch, sarvam_job_store

        if len(segments) == 1:
            wav_path, offset = segments[0]
            utterances = self._process_segment(session_id, wav_path, offset, num_segments=1)
            return sorted(utterances, key=lambda u: u.start_time)

        now = datetime.utcnow().isoformat()
        sarvam_job_store.create_pending_segments(session_id, [str(p) for p, _ in segments], now)
        concurrency = settings.sarvam_batch_segment_concurrency
        logger.info(f"[STT] session={session_id}: {len(segments)} segment(s) to process, "
                    f"max {concurrency} concurrent")
        if self.p: self.p.log(f"Sarvam Batch STT → {len(segments)} segment(s), "
                               f"up to {concurrency} concurrent")

        all_utterances: list[Utterance] = []
        errors: list[str] = []
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            future_map = {
                ex.submit(self._process_segment, f"{session_id}::seg{i}", wav_path, offset, len(segments)): i
                for i, (wav_path, offset) in enumerate(segments)
            }
            for future in as_completed(future_map):
                seg_idx = future_map[future]
                try:
                    all_utterances.extend(future.result())
                except sarvam_batch.SttJobFailedError as exc:
                    errors.append(f"segment {seg_idx}: {exc}")
                    logger.error(f"[STT] session={session_id} segment {seg_idx} failed: {exc}")
                progress = sarvam_job_store.segment_progress(session_id)
                logger.info(f"[STT] session={session_id} progress: "
                            f"{progress['completed']}/{progress['total_segments']} completed, "
                            f"{progress['processing']} processing, {progress['failed']} failed")
                if self.p: self.p.log(
                    f"Sarvam Batch STT → {progress['completed']}/{progress['total_segments']} "
                    f"segments completed ({progress['failed']} failed)")

        if errors:
            raise sarvam_batch.SttJobFailedError(
                f"{len(errors)}/{len(segments)} segment(s) failed permanently: " + "; ".join(errors[:5])
            )

        return sorted(all_utterances, key=lambda u: u.start_time)

    def _process_segment(self, seg_session_id: str, wav_path: Path, offset: float,
                          num_segments: int) -> list[Utterance]:
        """
        Full per-segment lifecycle: cached-transcript short-circuit ->
        smart-resume (live status check on a prior job) -> initialise ->
        upload -> start -> poll -> download -> parse -> cache -> build
        Utterances. Safe to call concurrently for different seg_session_ids
        (each only ever touches its own DB row; no shared mutable state).
        """
        import json as _json
        import dataclasses
        from transcription import sarvam_batch, sarvam_job_store

        now = datetime.utcnow().isoformat()
        prior = sarvam_job_store.get(seg_session_id)

        # Zero-cost skip: a previously COMPLETED segment's parsed transcript
        # is cached in the DB — never contact Sarvam again for it, not even
        # a status check. This is the strongest form of "never retranscribe
        # a completed chunk" — it holds even if Sarvam later expires/evicts
        # the job on its own side.
        if prior and prior.get("status") == "completed" and prior.get("transcript_json"):
            logger.info(f"[STT] session={seg_session_id}: cached transcript found — "
                        f"skipping Sarvam entirely (0 calls)")
            cached = _json.loads(prior["transcript_json"])
            result = sarvam_batch.BatchTranscriptResult(
                transcript=cached.get("transcript", ""),
                language_code=cached.get("language_code", "unknown"),
                has_diarization=cached.get("has_diarization", False),
                segments=[sarvam_batch.DiarizedSegment(**s) for s in cached.get("segments", [])],
            )
            return self._utterances_from_result(result, wav_path, offset)

        job = None
        skip_submit = False
        if prior and prior.get("sarvam_job_id"):
            live_state = sarvam_batch.resolve_existing_job_state(prior["sarvam_job_id"])
            if live_state == "Completed":
                logger.info(f"[STT] session={seg_session_id} job={prior['sarvam_job_id']} "
                            f"already Completed — resuming without resubmission")
                if self.p: self.p.log(f"Sarvam job {prior['sarvam_job_id']} already completed — "
                                       f"reusing (no resubmission, no re-pay)")
                job = sarvam_batch.get_job_handle(prior["sarvam_job_id"])
                skip_submit = True
            elif live_state in ("Accepted", "Pending", "Running"):
                logger.info(f"[STT] session={seg_session_id} job={prior['sarvam_job_id']} "
                            f"still {live_state} — resuming poll without resubmission")
                if self.p: self.p.log(f"Sarvam job {prior['sarvam_job_id']} still {live_state} — "
                                       f"resuming poll (no resubmission)")
                job = sarvam_batch.get_job_handle(prior["sarvam_job_id"])
                skip_submit = True
                sarvam_job_store.update_status(seg_session_id, "started", now=now)

        if not skip_submit:
            seg_duration = self.processor.get_duration(wav_path)
            try:
                job = sarvam_batch.submit_job(wav_path)
            except Exception as exc:
                logger.error(f"[STT] session={seg_session_id} initialise failed: {exc}")
                sarvam_job_store.upsert_submitted(seg_session_id, "", str(wav_path), seg_duration,
                                                   num_segments, now)
                sarvam_job_store.update_status(seg_session_id, "stt_failed", str(exc), now=now)
                raise sarvam_batch.SttJobFailedError(f"Sarvam job initialisation failed: {exc}") from exc

            sarvam_job_store.upsert_submitted(seg_session_id, job.job_id, str(wav_path),
                                               seg_duration, num_segments, now)
            logger.info(f"[STT] session={seg_session_id} job={job.job_id} submitted "
                        f"(duration={seg_duration:.0f}s)")
            if self.p: self.p.log(f"Sarvam Batch STT → Job submitted (job_id={job.job_id})")

            try:
                sarvam_batch.upload_and_start(job, wav_path)
                sarvam_job_store.update_status(seg_session_id, "started",
                                                now=datetime.utcnow().isoformat())
            except Exception as exc:
                logger.error(f"[STT] session={seg_session_id} job={job.job_id} upload/start failed: {exc}")
                sarvam_job_store.update_status(seg_session_id, "stt_failed", str(exc),
                                                now=datetime.utcnow().isoformat())
                raise sarvam_batch.SttJobFailedError(
                    f"Sarvam upload/start failed for job {job.job_id}: {exc}") from exc

        if self.p: self.p.log("Sarvam Batch STT → Processing")

        def _on_poll(status, _sid=seg_session_id):
            sarvam_job_store.update_status(_sid, "polling", now=datetime.utcnow().isoformat())
            logger.info(f"[STT] session={_sid} job={status.job_id} status={status.job_state} "
                        f"total_files={getattr(status, 'total_files', None)} "
                        f"ok={getattr(status, 'successful_files_count', None)} "
                        f"failed={getattr(status, 'failed_files_count', None)}")

        def _is_cancelled(_sid=seg_session_id):
            return sarvam_job_store.is_cancel_requested(_sid)

        try:
            final_state = sarvam_batch.poll_with_backoff(
                job, seg_session_id, on_status=_on_poll, is_cancelled=_is_cancelled)
        except sarvam_batch.SttJobFailedError as exc:
            sarvam_job_store.update_status(seg_session_id, "stt_failed", str(exc),
                                            now=datetime.utcnow().isoformat())
            logger.error(f"[STT] session={seg_session_id} timed out waiting on Sarvam: {exc}")
            raise

        if final_state == "Cancelled":
            sarvam_job_store.update_status(seg_session_id, "cancelled",
                                            now=datetime.utcnow().isoformat())
            logger.info(f"[STT] session={seg_session_id} cancelled — job {job.job_id} "
                        f"keeps running on Sarvam's side; Resume will pick up its result")
            if self.p: self.p.log("Transcription cancelled")
            raise sarvam_batch.SttJobFailedError(f"Cancelled by user (job {job.job_id} still running on Sarvam)")

        if final_state == "Failed":
            try:
                file_results = job.get_file_results()
                err = "; ".join(
                    (f.get("error_message") or "unknown error") for f in file_results.get("failed", [])
                ) or "Sarvam job failed"
            except Exception:
                err = "Sarvam job failed"
            sarvam_job_store.update_status(seg_session_id, "stt_failed", err,
                                            now=datetime.utcnow().isoformat())
            logger.error(f"[STT] session={seg_session_id} job={job.job_id} failed: {err}")
            raise sarvam_batch.SttJobFailedError(f"Sarvam job {job.job_id} failed: {err}")

        if self.p: self.p.log("Sarvam Batch STT → Completed")

        out_dir = wav_path.parent / "sarvam_batch_out"
        try:
            result = sarvam_batch.download_and_parse(job, wav_path, out_dir)
        except sarvam_batch.SttJobFailedError as exc:
            sarvam_job_store.update_status(seg_session_id, "stt_failed", str(exc),
                                            now=datetime.utcnow().isoformat())
            logger.error(f"[STT] session={seg_session_id} job={job.job_id} download/parse failed: {exc}")
            raise

        cache_payload = _json.dumps({
            "transcript": result.transcript,
            "language_code": result.language_code,
            "has_diarization": result.has_diarization,
            "segments": [dataclasses.asdict(s) for s in result.segments],
        })
        sarvam_job_store.save_transcript(seg_session_id, cache_payload, now=datetime.utcnow().isoformat())
        logger.info(f"[STT] session={seg_session_id} job={job.job_id} completed "
                    f"({len(result.segments)} segments, diarized={result.has_diarization})")
        if self.p: self.p.log(f"Sarvam Batch STT → Transcript ready "
                               f"({len(result.segments)} segments, diarized={result.has_diarization})")

        return self._utterances_from_result(result, wav_path, offset)

    def _utterances_from_result(self, result, wav_path: Path, offset: float) -> list[Utterance]:
        """Speaker-role assignment + Utterance construction for one parsed
        segment's BatchTranscriptResult, shared by the live-fetch and
        cached-transcript paths in _process_segment."""
        from diarization.diarizer import assign_roles_by_talktime
        from config.models import AudioChunk

        chunk_role = None
        if result.has_diarization:
            durations: dict[str, float] = {}
            for seg in result.segments:
                durations[seg.speaker_id] = durations.get(seg.speaker_id, 0.0) + max(0.0, seg.end - seg.start)
            role_map = assign_roles_by_talktime(durations)
        else:
            # Fallback: existing acoustic diarizer (stereo/mono-energy), run
            # directly against the segment wav using the timestamped-chunk
            # boundaries as its "chunks" input — no physical re-splitting
            # needed (read_wav_rms reads time ranges directly from the file).
            role_map = None
            synth_chunks = [
                AudioChunk(chunk_id=i, start_time=seg.start, end_time=seg.end, file_path=str(wav_path))
                for i, seg in enumerate(result.segments)
            ]
            chunk_role = self.diarizer.diarize(wav_path, synth_chunks)

        utterances: list[Utterance] = []
        for i, seg in enumerate(result.segments):
            speaker = (role_map.get(seg.speaker_id, Speaker.UNKNOWN) if role_map is not None
                       else chunk_role.get(i, Speaker.UNKNOWN))
            language_detected = result.language_code if result.language_code != "unknown" else "en-IN"
            english_text = (
                seg.text if language_detected.startswith("en")
                else self.stt._translate_to_english(seg.text, language_detected)
            )
            utterances.append(Utterance(
                utterance_id=str(uuid.uuid4()),
                speaker=speaker,
                start_time=round(seg.start + offset, 3),
                end_time=round(seg.end + offset, 3),
                native_text=seg.text,
                english_text=english_text,
                language_detected=language_detected,
                confidence=1.0,
            ))
        return utterances

    # ── Parallel STT (legacy — 28s-chunk path, kept for transcription/stt.py's
    # per-chunk cascade; no longer called by run() by default, see _batch_stt) ──

    def _parallel_stt(self, chunks) -> list[Utterance]:
        results: list[Utterance] = []
        with ThreadPoolExecutor(max_workers=STT_WORKERS) as ex:
            future_map = {
                ex.submit(self.stt._transcribe_chunk, chunk, Speaker.UNKNOWN): chunk
                for chunk in chunks
            }
            for future in as_completed(future_map):
                chunk = future_map[future]
                try:
                    utt = future.result()
                    if utt and utt.native_text.strip():
                        results.append(utt)
                        if self.p:
                            self.p.chunk_ok(chunk.chunk_id, utt.language_detected,
                                            utt.native_text, utt.english_text)
                    else:
                        if self.p: self.p.chunk_fail(chunk.chunk_id, "empty")
                except Exception as exc:
                    logger.warning(f"Chunk {chunk.chunk_id} failed: {exc}")
                    if self.p: self.p.chunk_fail(chunk.chunk_id, str(exc)[:80])
        return sorted(results, key=lambda u: u.start_time)

    # ── Setup utterance filter ────────────────────────────────────────────────

    @staticmethod
    def _strip_setup_utterances(utterances) -> list:
        """
        Remove the opening audio/video setup exchanges that contain no real
        session content (e.g. 'Hello, can you hear me?', 'Am I audible?').

        Strategy:
          1. Look at utterances before the 3-minute mark.
          2. Flag any utterance whose English text matches common setup patterns.
          3. Find the first utterance after the setup exchange that carries real
             content (len > 15 words) — strip everything before it only if
             the stripped portion is genuinely setup talk.
          4. Safety cap: never strip more than the first 8 utterances.
        """
        import re
        SETUP_PATTERNS = [
            r"\bam i audible\b", r"\bcan you hear me\b", r"\bcan you see me\b",
            r"\bhello.*hello\b", r"\bhi.*hi\b", r"\baudio.*check\b",
            r"\bvideo.*check\b", r"\bturn on.*audio\b", r"\bturn on.*video\b",
            r"\bplease.*turn.*on\b", r"\bmike.*on\b", r"\bmic.*on\b",
            r"\bjoin.*audio\b", r"\bjoin.*video\b",
            r"\bcheck.*one.*two\b", r"\bcheck.*check\b",
            r"\bnetwork.*issue\b", r"\bcoming.*through\b",
            r"\bsignal.*outside\b", r"\binternet.*issue\b",
            r"\bscreen.*share\b", r"\bshare.*screen\b",
            r"^(hello[,\s!.]*)+$", r"^(hi[,\s!.]*)+$",
        ]

        MAX_STRIP = 8
        SETUP_CUTOFF_SEC = 180   # only consider first 3 minutes

        def _is_setup(utt) -> bool:
            text = (utt.english_text or "").lower().strip()
            if not text:
                return True
            # Very short (≤5 words) inside first 3 min
            if utt.start_time < SETUP_CUTOFF_SEC and len(text.split()) <= 5:
                return True
            for pat in SETUP_PATTERNS:
                if re.search(pat, text, re.IGNORECASE):
                    return True
            return False

        # Find first "real" utterance
        first_real = 0
        for i, utt in enumerate(utterances[:MAX_STRIP]):
            if _is_setup(utt):
                first_real = i + 1
            else:
                # Non-setup found — stop scanning
                break

        if first_real == 0:
            return utterances  # nothing to strip

        logger.info(f"Stripped {first_real} setup utterance(s) from transcript start")
        return utterances[first_real:]

    # ── Speaker label injection ───────────────────────────────────────────────

    @staticmethod
    def _inject_speaker_labels(utterances, chunks, speaker_map):
        for utt in utterances:
            best = min(chunks, key=lambda c: abs(c.start_time - utt.start_time), default=None)
            if best is not None:
                utt.speaker = speaker_map.get(best.chunk_id, Speaker.UNKNOWN)
