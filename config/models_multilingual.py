"""
config/models_multilingual.py
Pydantic data models for the multilingual sales call audit pipeline.
Covers all 9 pipeline layers from audio input to structured JSON output.
"""
from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────────────────────

class SpeakerRole(str, Enum):
    SALESPERSON = "Salesperson"
    CUSTOMER    = "Customer"
    UNKNOWN     = "Unknown"


class Language(str, Enum):
    ENGLISH    = "en"
    HINDI      = "hi"
    HINGLISH   = "hinglish"
    KANNADA    = "kn"
    TAMIL      = "ta"
    TELUGU     = "te"
    MALAYALAM  = "ml"
    BENGALI    = "bn"
    MARATHI    = "mr"
    GUJARATI   = "gu"
    PUNJABI    = "pa"
    UNKNOWN    = "unknown"


class DealRisk(str, Enum):
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"


# ── Layer 1: Audio ─────────────────────────────────────────────────────────────

class VADSegment(BaseModel):
    start: float
    end:   float
    speech_prob: float


class AudioChunkV2(BaseModel):
    chunk_id:     int
    start_time:   float
    end_time:     float
    file_path:    str
    speech_ratio: float = 1.0       # fraction of chunk that is speech
    energy_db:    float = -20.0


# ── Layer 2: Language Detection ────────────────────────────────────────────────

class LangSegment(BaseModel):
    start:      float
    end:        float
    language:   Language
    confidence: float


class LanguageMap(BaseModel):
    primary_language:     Language
    code_switch_density:  float = 0.0      # 0–1, fraction of time spent in non-primary lang
    segments:             list[LangSegment] = Field(default_factory=list)
    languages_detected:   list[Language]   = Field(default_factory=list)


# ── Layer 3: Speaker Diarization ───────────────────────────────────────────────

class SpeakerSegment(BaseModel):
    speaker:    SpeakerRole
    start:      float
    end:        float
    duration:   float


class TalkAnalytics(BaseModel):
    salesperson_pct:        float
    customer_pct:           float
    salesperson_duration:   float      # seconds
    customer_duration:      float
    interruption_count:     int
    longest_monologue_sec:  float
    avg_response_latency:   float
    dead_air_sec:           float
    questions_asked:        int


# ── Layer 4 + 5: Transcription & Post-Processing ──────────────────────────────

class WordToken(BaseModel):
    word:       str
    start:      float
    end:        float
    speaker:    SpeakerRole = SpeakerRole.UNKNOWN
    confidence: float = 1.0


class Utterance(BaseModel):
    utterance_id:     str
    speaker:          SpeakerRole
    start_time:       float
    end_time:         float
    text:             str           # original language, punctuated
    english_text:     str           # translated if needed
    language:         Language
    confidence:       float = 1.0
    words:            list[WordToken] = Field(default_factory=list)
    is_question:      bool = False


# ── Layer 6: Semantic Chunking ─────────────────────────────────────────────────

class TopicSegment(BaseModel):
    segment_id:    int
    title:         str
    start_time:    float
    end_time:      float
    utterance_ids: list[str]
    summary:       str = ""


# ── Layer 7: Call Intelligence ─────────────────────────────────────────────────

class EvidenceQuote(BaseModel):
    text:      str
    speaker:   SpeakerRole
    timestamp: str     # HH:MM:SS
    language:  Language = Language.ENGLISH


class PainPoint(BaseModel):
    description: str
    severity:    str   # "high" | "medium" | "low"
    quote:       EvidenceQuote


class BudgetSignal(BaseModel):
    mentioned:    bool
    amount:       Optional[str] = None
    sentiment:    str = "neutral"    # "positive" | "negative" | "neutral"
    quote:        Optional[EvidenceQuote] = None


class Objection(BaseModel):
    type:         str          # "price" | "timing" | "authority" | "need" | "trust"
    text:         str
    handled:      bool
    handling_quality: Optional[int] = None   # 1–10
    quote:        EvidenceQuote


class BuyingSignal(BaseModel):
    signal_type: str    # "question_about_next_steps" | "positive_language" | "timeline_mention"
    text:        str
    strength:    str    # "strong" | "moderate" | "weak"
    quote:       EvidenceQuote


class CallIntelligence(BaseModel):
    pain_points:        list[PainPoint]    = Field(default_factory=list)
    budget_signals:     list[BudgetSignal] = Field(default_factory=list)
    timeline:           Optional[str]      = None
    decision_makers:    list[str]          = Field(default_factory=list)
    objections:         list[Objection]    = Field(default_factory=list)
    competitors:        list[str]          = Field(default_factory=list)
    next_steps:         list[str]          = Field(default_factory=list)
    buying_signals:     list[BuyingSignal] = Field(default_factory=list)
    risk_signals:       list[str]          = Field(default_factory=list)
    deal_risk:          DealRisk           = DealRisk.MEDIUM


# ── Layer 8: Sales Audit ───────────────────────────────────────────────────────

class AuditDimension(BaseModel):
    name:       str
    score:      int                  # 1–10
    weight:     float                # 0–1, contribution to total
    rationale:  str
    evidence:   list[EvidenceQuote]  = Field(default_factory=list)
    positives:  list[str]            = Field(default_factory=list)
    negatives:  list[str]            = Field(default_factory=list)


class SalesAuditScore(BaseModel):
    opening:            AuditDimension
    rapport_building:   AuditDimension
    discovery:          AuditDimension
    requirement_gathering: AuditDimension
    product_understanding: AuditDimension
    objection_handling: AuditDimension
    value_communication: AuditDimension
    closing:            AuditDimension
    follow_up_clarity:  AuditDimension

    total_score:  float   # weighted 0–100
    grade:        str     # A+/A/B/C/D/F
    rubric_version: str   = "v1.0"


# ── Layer 9: Coaching Engine ───────────────────────────────────────────────────

class CoachingFinding(BaseModel):
    category:    str     # "strength" | "weakness" | "missed_opportunity"
    title:       str
    detail:      str
    timestamp:   Optional[str] = None
    quote:       Optional[EvidenceQuote] = None


class SuggestedResponse(BaseModel):
    context:        str     # what the rep said
    what_was_said:  str     # verbatim or paraphrased
    better_response: str    # ideal alternative
    why:            str     # coaching rationale


class CoachingReport(BaseModel):
    strengths:            list[CoachingFinding]   = Field(default_factory=list)
    weaknesses:           list[CoachingFinding]   = Field(default_factory=list)
    missed_opportunities: list[CoachingFinding]   = Field(default_factory=list)
    suggested_responses:  list[SuggestedResponse] = Field(default_factory=list)
    recommended_actions:  list[str]               = Field(default_factory=list)
    summary:              str                     = ""


# ── Full Output ────────────────────────────────────────────────────────────────

class CallAuditReport(BaseModel):
    session_id:         str
    recording_file:     str
    duration_seconds:   float
    processed_at:       str

    # Layer 2
    language_map:       LanguageMap

    # Layer 3
    talk_analytics:     TalkAnalytics
    speaker_segments:   list[SpeakerSegment]  = Field(default_factory=list)

    # Layer 4+5
    utterances:         list[Utterance]        = Field(default_factory=list)
    call_summary:       str                    = ""

    # Layer 6
    topic_segments:     list[TopicSegment]     = Field(default_factory=list)

    # Layer 7
    call_intelligence:  CallIntelligence

    # Layer 8
    sales_audit:        SalesAuditScore

    # Layer 9
    coaching_report:    CoachingReport

    # Meta
    processing_time_sec: float = 0.0
    models_used:        dict   = Field(default_factory=dict)
