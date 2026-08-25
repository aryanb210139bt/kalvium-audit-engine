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

    def run(self, recording_path: str | Path) -> dict:
        """
        Returns a dict (DemoAuditReport.model_dump() + advanced_sentiment + category_coaching)
        so that reports can be generated directly from it.
        """
        recording_path = Path(recording_path)
        session_id     = str(uuid.uuid4())
        started_at     = datetime.utcnow()

        logger.info(f"Pipeline V3 start — {recording_path.name}")
        if self.p: self.p.log(f"Starting V3 audit: {recording_path.name}")

        # ── Step 1 & 2: Audio → WAV → chunks ─────────────────────────────────
        if self.p: self.p.step_start(1, f"Converting {recording_path.name}")
        wav_path, chunks = self.processor.process(recording_path)
        duration = self.processor.get_duration(wav_path)
        if self.p:
            self.p.chunks_total = len(chunks)
            self.p.step_done(1, f"{duration:.0f}s audio")
            self.p.step_start(2, f"{len(chunks)} chunks (≤25s each)")
            self.p.step_done(2, f"{len(chunks)} chunks ready")

        # ── Step 3: Parallel STT ──────────────────────────────────────────────
        if self.p: self.p.step_start(3, f"{len(chunks)} chunks → Sarvam AI ({STT_WORKERS}×)")
        utterances = self._parallel_stt(chunks)
        if self.p: self.p.step_done(3, f"{len(utterances)}/{len(chunks)} transcribed")

        # ── Step 4: Diarize + inject labels ───────────────────────────────────
        if self.p: self.p.step_start(4, "Assigning speaker labels")
        speaker_map = self.diarizer.diarize(wav_path, chunks)
        self._inject_speaker_labels(utterances, chunks, speaker_map)
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

    # ── Parallel STT ──────────────────────────────────────────────────────────

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
