"""
config/models.py
Pydantic data models shared across the entire pipeline.
"""
from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────────────────────

class Speaker(str, Enum):
    COUNSELLOR = "Counsellor"
    STUDENT = "Student"
    PARENT = "Parent"
    UNKNOWN = "Unknown"


class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class IntentType(str, Enum):
    HIGH_INTEREST = "high_interest"
    LOW_INTEREST = "low_interest"
    CONFUSED = "confused"
    FEE_CONCERN = "fee_concern"
    PLACEMENT_CONCERN = "placement_concern"
    CREDIBILITY_CONCERN = "credibility_concern"
    PARENT_RESISTANCE = "parent_resistance"
    SCHOLARSHIP_INTEREST = "scholarship_interest"
    STUDENT_EXCITED = "student_excited"
    STUDENT_PASSIVE = "student_passive"
    PARENT_SKEPTICAL = "parent_skeptical"
    NEUTRAL = "neutral"


class ConversationEvent(str, Enum):
    COUNSELLOR_INTRO = "counsellor_intro"
    KALVIUM_EXPLANATION = "kalvium_explanation"
    TRADITIONAL_COLLEGE_COMPARISON = "traditional_college_comparison"
    CAREER_OUTCOMES = "career_outcomes"
    REPO_SHOWN = "repo_shown"
    PPT_PRESENTED = "ppt_presented"
    PLACEMENTS_DISCUSSION = "placements_discussion"
    FEE_STRUCTURE = "fee_structure"
    ROI_EXPLANATION = "roi_explanation"
    DISCOVERY_QUESTIONS = "discovery_questions"
    CLOSING_ATTEMPT = "closing_attempt"
    OBJECTION_RAISED = "objection_raised"
    OBJECTION_HANDLED = "objection_handled"


# ── Pipeline models ───────────────────────────────────────────────────────────

class AudioChunk(BaseModel):
    chunk_id: int
    start_time: float          # seconds
    end_time: float
    file_path: str
    speaker: Speaker = Speaker.UNKNOWN


class Utterance(BaseModel):
    utterance_id: str
    speaker: Speaker
    start_time: float
    end_time: float
    native_text: str           # original language
    english_text: str          # translated
    language_detected: str = "en"
    confidence: float = 1.0


class StructuredEvent(BaseModel):
    """Core atomic unit fed into the audit engine."""
    utterance_id: str
    speaker: Speaker
    event: ConversationEvent
    timestamp: str             # HH:MM:SS
    sentiment: Sentiment
    native_text: str
    english_text: str
    confidence: float = 1.0


# ── Audit result models ───────────────────────────────────────────────────────

class ScriptComplianceResult(BaseModel):
    counsellor_intro: bool = False
    kalvium_explanation: bool = False
    traditional_comparison: bool = False
    career_outcomes: bool = False
    repo_shown: bool = False
    ppt_presented: bool = False
    placements_discussed: bool = False
    fee_structure_explained: bool = False
    roi_explained: bool = False
    discovery_questions_asked: bool = False
    closing_attempted: bool = False

    @property
    def completion_rate(self) -> float:
        fields = list(self.model_fields.keys())
        completed = sum(1 for f in fields if getattr(self, f))
        return round(completed / len(fields) * 100, 1)


class TalkRatioResult(BaseModel):
    counsellor_pct: float
    student_pct: float
    parent_pct: float
    total_questions_asked: int
    counsellor_questions: int
    interruption_count: int
    avg_response_latency_sec: float
    dead_air_sec: float
    is_balanced: bool          # True if counsellor < 70%

    @property
    def quality_label(self) -> str:
        if self.counsellor_pct > 80:
            return "Counsellor dominated — needs improvement"
        elif self.counsellor_pct > 65:
            return "Slightly unbalanced"
        return "Well balanced"


class IntentResult(BaseModel):
    student_intent: IntentType
    parent_intent: IntentType
    overall_intent: IntentType
    alignment_risk: str        # "low" | "medium" | "high"
    key_concerns: list[str] = Field(default_factory=list)
    admission_probability: float = 0.0   # 0–1


class SentimentResult(BaseModel):
    overall: Sentiment
    counsellor_sentiment: Sentiment
    student_sentiment: Sentiment
    parent_sentiment: Sentiment
    sentiment_timeline: list[dict] = Field(default_factory=list)


class ObjectionResult(BaseModel):
    objection_text: str
    category: str
    timestamp: str
    was_addressed: bool
    resolution_quality: str    # "well", "partial", "missed"


class ObjectionAnalysis(BaseModel):
    total_objections: int
    resolved: int
    partially_resolved: int
    missed: int
    objections: list[ObjectionResult] = Field(default_factory=list)

    @property
    def handling_rate(self) -> float:
        if self.total_objections == 0:
            return 100.0
        return round((self.resolved / self.total_objections) * 100, 1)


class LLMEvaluationResult(BaseModel):
    """Result from the OpenAI LLM audit layer."""
    category: str
    score: float               # 0–10
    reasoning: str
    evidence: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


class CategoryScore(BaseModel):
    category: str
    weight: float              # percentage weight in final score
    raw_score: float           # 0–10
    weighted_score: float
    breakdown: str


class AuditScore(BaseModel):
    overall: float             # 0–100
    grade: str                 # A / B / C / D / F
    product_explanation: CategoryScore
    discovery_questions: CategoryScore
    engagement: CategoryScore
    objection_handling: CategoryScore
    parent_alignment: CategoryScore
    confidence_clarity: CategoryScore
    closing_skills: CategoryScore
    demo_completeness: CategoryScore


class DemoAuditReport(BaseModel):
    """Final output of the entire pipeline."""
    session_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Input metadata
    recording_file: str
    duration_seconds: float
    language_detected: str
    speakers_detected: list[str]

    # Pipeline outputs
    utterances: list[Utterance] = Field(default_factory=list)
    structured_events: list[StructuredEvent] = Field(default_factory=list)

    # Intelligence outputs
    script_compliance: ScriptComplianceResult
    talk_ratio: TalkRatioResult
    intent: IntentResult
    sentiment: SentimentResult
    objections: ObjectionAnalysis
    llm_evaluations: list[LLMEvaluationResult] = Field(default_factory=list)

    # Final score
    score: AuditScore

    # Coaching
    coaching_highlights: list[str] = Field(default_factory=list)
    top_strengths: list[str] = Field(default_factory=list)
    improvement_areas: list[str] = Field(default_factory=list)
