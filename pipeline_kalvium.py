"""
pipeline_kalvium.py
Kalvium Webinar Registration Authenticity Pipeline.

5-Layer AI analysis with correct philosophy:
  NOT sales conversion — webinar attendance authenticity detection.

Flow:
  Audio → WAV → chunks(≤28s) → Sarvam STT → diarize → labeled transcript
  → [L1 compliance | L2 engagement | L3 registration] parallel
  → L4 attendance → L5 fraud → score → report
"""
from __future__ import annotations
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from config.models import Speaker, Utterance, TalkRatioResult
from config.kalvium_models import KalviumAuditReport
from config.settings import get_settings

from audio.processor import AudioProcessor
from transcription.stt import SpeechToText
from diarization.diarizer import Diarizer

from audit.kalvium_layers import (
    analyze_compliance,
    analyze_engagement,
    analyze_registration,
    analyze_attendance,
    analyze_fraud,
)
from scoring.fraud_scorer import FraudScorer

logger   = logging.getLogger(__name__)
settings = get_settings()

STT_WORKERS   = 4   # parallel Sarvam API calls
LAYER_WORKERS = 3   # layers 1-3 run in parallel


def _build_transcript(utterances: list[Utterance]) -> str:
    """[MM:SS|A] text  or  [MM:SS|S] text  or  [MM:SS|P] text"""
    role_map = {
        Speaker.COUNSELLOR: "A",   # Associate
        Speaker.STUDENT:    "S",   # Student
        Speaker.PARENT:     "P",   # Parent
        Speaker.UNKNOWN:    "S",   # default to student
    }
    lines = []
    for u in utterances:
        mins = int(u.start_time // 60)
        secs = int(u.start_time  % 60)
        role = role_map.get(u.speaker, "S")
        lines.append(f"[{mins:02d}:{secs:02d}|{role}] {u.english_text}")
    return "\n".join(lines)


def _talk_ratios(utterances: list[Utterance], duration: float) -> dict[str, float]:
    secs: dict[str, float] = {
        "associate": 0, "student": 0, "parent": 0
    }
    for u in utterances:
        d = u.end_time - u.start_time
        if u.speaker == Speaker.COUNSELLOR:
            secs["associate"] += d
        elif u.speaker == Speaker.STUDENT:
            secs["student"] += d
        elif u.speaker == Speaker.PARENT:
            secs["parent"] += d
        else:
            secs["student"] += d   # UNKNOWN → student

    base = max(sum(secs.values()), 1)
    return {k: round(v / base * 100, 1) for k, v in secs.items()}


def _make_talk_ratio(utterances, duration) -> TalkRatioResult:
    tr = _talk_ratios(utterances, duration)
    spoken = sum(u.end_time - u.start_time for u in utterances)
    dead_air = max(0.0, duration - spoken)
    c_pct = tr["associate"]
    return TalkRatioResult(
        counsellor_pct        = c_pct,
        student_pct           = tr["student"],
        parent_pct            = tr["parent"],
        total_questions_asked = sum(1 for u in utterances if "?" in u.english_text),
        counsellor_questions  = sum(
            1 for u in utterances
            if u.speaker == Speaker.COUNSELLOR and "?" in u.english_text
        ),
        interruption_count       = 0,
        avg_response_latency_sec = 0.0,
        dead_air_sec             = round(dead_air, 1),
        is_balanced              = c_pct < 70,
    )


class KalviumAuditPipeline:

    def __init__(self):
        self.processor = AudioProcessor()
        self.stt       = SpeechToText()
        self.diarizer  = Diarizer()
        self.scorer    = FraudScorer()

    def run(
        self,
        recording_path: str | Path,
        associate_id:   str = "",
        associate_name: str = "",
        lead_id:        str = "",
        prospect_name:  str = "",
    ) -> KalviumAuditReport:
        recording_path = Path(recording_path)
        audit_id       = str(uuid.uuid4())
        started_at     = datetime.utcnow()

        logger.info(f"[{audit_id}] Pipeline start → {recording_path.name}")

        # ── 1. Audio processing ───────────────────────────────────────────────
        logger.info(f"[{audit_id}] Step 1: Audio → WAV → ≤28s chunks")
        wav_path, chunks = self.processor.process(recording_path)
        duration         = self.processor.get_duration(wav_path)
        logger.info(f"[{audit_id}] {duration:.0f}s, {len(chunks)} chunks")

        # ── 2. Diarization (before STT — tells us who speaks which chunk) ─────
        logger.info(f"[{audit_id}] Step 2: Speaker diarization")
        speaker_map = self.diarizer.diarize(wav_path, chunks)

        # ── 3. Parallel STT (Sarvam) ──────────────────────────────────────────
        logger.info(f"[{audit_id}] Step 3: Sarvam STT ({len(chunks)} chunks, {STT_WORKERS} parallel)")
        utterances = self._parallel_stt(chunks, speaker_map, audit_id)
        logger.info(f"[{audit_id}] {len(utterances)} utterances transcribed")

        if not utterances:
            logger.warning(f"[{audit_id}] No utterances — silent/failed audio")

        # ── 4. Transcript assembly ────────────────────────────────────────────
        language     = utterances[0].language_detected if utterances else "hi-IN"
        transcript   = _build_transcript(utterances)
        tr           = _talk_ratios(utterances, duration)
        talk_ratio   = _make_talk_ratio(utterances, duration)

        logger.info(
            f"[{audit_id}] Talk: Associate={tr['associate']}% "
            f"Student={tr['student']}% Parent={tr['parent']}%"
        )

        # ── 5. Layers 1-3 in parallel ─────────────────────────────────────────
        logger.info(f"[{audit_id}] Step 4: Layers 1-3 (parallel)")
        with ThreadPoolExecutor(max_workers=LAYER_WORKERS) as ex:
            f1 = ex.submit(analyze_compliance,  utterances, transcript)
            f2 = ex.submit(analyze_engagement,  utterances, transcript, tr["associate"])
            f3 = ex.submit(analyze_registration, utterances, transcript, duration, tr["associate"])
            compliance  = f1.result()
            engagement  = f2.result()
            registration= f3.result()

        # ── 6. Layer 4 — Attendance (needs L1-L3) ────────────────────────────
        logger.info(f"[{audit_id}] Step 5: Layer 4 — Attendance prediction")
        attendance = analyze_attendance(
            utterances, transcript, compliance, engagement, registration
        )

        # ── 7. Layer 5 — Fraud detection (needs L1-L3) ───────────────────────
        logger.info(f"[{audit_id}] Step 6: Layer 5 — Fraud detection")
        fraud = analyze_fraud(
            utterances, transcript, duration,
            tr["associate"], tr["student"],
            compliance, registration,
        )

        # ── 8. Assemble report ────────────────────────────────────────────────
        elapsed = (datetime.utcnow() - started_at).total_seconds()
        report  = KalviumAuditReport(
            audit_id          = audit_id,
            associate_id      = associate_id,
            associate_name    = associate_name,
            lead_id           = lead_id,
            prospect_name     = prospect_name,
            call_date         = started_at.strftime("%Y-%m-%d"),
            recording_file    = str(recording_path),
            duration_seconds  = duration,
            language_detected = language,
            talk_ratio_associate = tr["associate"],
            talk_ratio_prospect  = tr["student"],
            talk_ratio_parent    = tr["parent"],
            silence_ratio        = max(0, 100 - tr["associate"] - tr["student"] - tr["parent"]),
            compliance    = compliance,
            engagement    = engagement,
            registration  = registration,
            attendance    = attendance,
            fraud         = fraud,
        )

        # ── 9. Score + manager summary ────────────────────────────────────────
        logger.info(f"[{audit_id}] Step 7: Scoring + manager summary")
        report = self.scorer.score(report)

        logger.info(
            f"[{audit_id}] DONE in {elapsed:.0f}s → "
            f"{report.final_classification.value} | "
            f"fake={report.fake_probability:.0%} | "
            f"attend={report.attendance_probability:.0%}"
        )
        return report

    def _parallel_stt(self, chunks, speaker_map, audit_id) -> list[Utterance]:
        results: list[Utterance] = []
        with ThreadPoolExecutor(max_workers=STT_WORKERS) as ex:
            futures = {
                ex.submit(
                    self.stt._transcribe_chunk,
                    chunk,
                    speaker_map.get(chunk.chunk_id, Speaker.UNKNOWN),
                ): chunk
                for chunk in chunks
            }
            for fut in as_completed(futures):
                try:
                    utt = fut.result()
                    if utt and utt.native_text.strip():
                        results.append(utt)
                except Exception as exc:
                    logger.warning(f"[{audit_id}] STT chunk failed: {exc}")
        return sorted(results, key=lambda u: u.start_time)
