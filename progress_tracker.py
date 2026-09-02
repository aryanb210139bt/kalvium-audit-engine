"""
progress_tracker.py
Thread-safe progress event bridge: pipeline (sync thread) → WebSocket (async).
Pipeline writes events; WebSocket handler reads and pushes to browser.
"""
from __future__ import annotations
import queue
import threading
from datetime import datetime


class ProgressTracker:
    """
    One instance per audit session.
    All methods are safe to call from any thread.
    """

    STEPS = {
        1: "Audio Conversion",
        2: "Batch Segmentation",
        3: "Sarvam Batch STT",
        4: "Speaker Labeling",
        5: "Event Detection",
        6: "GPT-4o Audit",
        7: "Scoring & Coaching",
    }

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._q: queue.Queue[dict | None] = queue.Queue(maxsize=2000)
        self._lock = threading.Lock()
        self.history: list[dict] = []  # for WebSocket reconnect replay

        # live counters (read by UI)
        self.chunks_total = 0
        self.chunks_done = 0
        self.chunks_failed = 0
        self.utterances_found = 0
        self.audit_cats_done = 0
        self.current_step = 0

    # ── Core emit ─────────────────────────────────────────────────────────────

    def emit(self, event_type: str, **data) -> None:
        event = {"type": event_type, "ts": datetime.utcnow().isoformat(), **data}
        with self._lock:
            self.history.append(event)
        try:
            self._q.put_nowait(event)
        except queue.Full:
            pass  # drop if consumer is slow — non-critical telemetry

    # ── Step helpers ──────────────────────────────────────────────────────────

    def step_start(self, step: int, detail: str = "") -> None:
        self.current_step = step
        self.emit("step_start", step=step, name=self.STEPS.get(step, ""), detail=detail)

    def step_done(self, step: int, result: str = "") -> None:
        self.emit("step_done", step=step, name=self.STEPS.get(step, ""), result=result)

    # ── Chunk helpers ─────────────────────────────────────────────────────────

    def chunk_ok(self, chunk_id: int, lang: str, native: str, english: str) -> None:
        with self._lock:
            self.chunks_done += 1
            self.utterances_found += 1
            done = self.chunks_done
        self.emit(
            "chunk_ok",
            chunk_id=chunk_id,
            done=done,
            total=self.chunks_total,
            pct=round(done / max(self.chunks_total, 1) * 100, 1),
            lang=lang,
            native=native[:120],
            english=english[:120],
        )

    def chunk_fail(self, chunk_id: int, reason: str) -> None:
        with self._lock:
            self.chunks_done += 1
            self.chunks_failed += 1
            done = self.chunks_done
        self.emit(
            "chunk_fail",
            chunk_id=chunk_id,
            done=done,
            total=self.chunks_total,
            reason=reason[:80],
        )

    # ── Audit helpers ─────────────────────────────────────────────────────────

    def audit_start(self, category: str) -> None:
        self.emit("audit_start", category=category)

    def audit_done(
        self,
        category: str,
        category_id: str,
        score: float,
        why_score: str = "",
        reason_for_zero: str = "",
        evidence_found: list | None = None,
        missing_behaviors: list | None = None,
        coaching_feedback: str = "",
        score_confidence: float = 1.0,
        evaluation_basis: str = "gpt",
    ) -> None:
        with self._lock:
            self.audit_cats_done += 1
        self.emit(
            "audit_done",
            category=category,
            category_id=category_id,
            score=round(score, 1),
            why_score=why_score[:300],
            reason_for_zero=reason_for_zero[:200],
            evidence_found=(evidence_found or [])[:3],
            missing_behaviors=(missing_behaviors or [])[:3],
            coaching_feedback=coaching_feedback[:250],
            score_confidence=round(score_confidence, 2),
            evaluation_basis=evaluation_basis,
            done=self.audit_cats_done,
            total=19,
        )

    # ── Final result ──────────────────────────────────────────────────────────

    def score_ready(
        self,
        overall: float,
        grade: str,
        scores: dict,
        highlights: list,
        strengths: list,
        improvements: list,
        explainable: list | None = None,
        composite: dict | None = None,
    ) -> None:
        self.emit(
            "score_ready",
            overall=overall,
            grade=grade,
            scores=scores,
            highlights=highlights,
            strengths=strengths,
            improvements=improvements,
            explainable=explainable or [],
            composite=composite or {},
        )

    def log(self, message: str, level: str = "info") -> None:
        self.emit("log", message=message, level=level)

    def finish(self) -> None:
        self.emit("done", message="Audit complete")
        self._q.put(None)  # sentinel → closes WebSocket stream

    def error(self, message: str) -> None:
        self.emit("error", message=message)
        self._q.put(None)

    # ── Consumer (called from async WebSocket handler) ────────────────────────

    def get_event(self, timeout: float = 1.0) -> dict | None | str:
        """Returns: event dict | None (stream done) | 'TIMEOUT'."""
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return "TIMEOUT"


class ProgressRegistry:
    """Singleton registry of all active session trackers."""

    _instance: ProgressRegistry | None = None
    _lock = threading.Lock()

    def __init__(self):
        self._trackers: dict[str, ProgressTracker] = {}

    @classmethod
    def get(cls) -> ProgressRegistry:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def create(self, session_id: str) -> ProgressTracker:
        t = ProgressTracker(session_id)
        self._trackers[session_id] = t
        return t

    def lookup(self, session_id: str) -> ProgressTracker | None:
        return self._trackers.get(session_id)

    def remove(self, session_id: str) -> None:
        self._trackers.pop(session_id, None)
