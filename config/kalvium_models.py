"""
config/kalvium_models.py
Pydantic models for Kalvium Webinar Registration Authenticity Platform.

Philosophy: NOT a sales evaluation system.
Goal: Detect fake registrations and predict genuine webinar attendance.
"""
from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class FinalClassification(str, Enum):
    GENUINE        = "GENUINE"
    LIKELY_GENUINE = "LIKELY GENUINE"
    WEAK           = "WEAK"
    SUSPICIOUS     = "SUSPICIOUS"
    HIGH_RISK      = "HIGH RISK"


class FraudRiskLevel(str, Enum):
    LOW    = "LOW"
    MEDIUM = "MEDIUM"
    HIGH   = "HIGH"


class AttendanceConfidence(str, Enum):
    HIGH   = "HIGH"
    MEDIUM = "MEDIUM"
    LOW    = "LOW"


class RecommendedAction(str, Enum):
    APPROVE         = "APPROVE"
    MONITOR         = "MONITOR"
    FLAG_FOR_REVIEW = "FLAG FOR REVIEW"
    INVESTIGATE     = "INVESTIGATE"
    ESCALATE        = "ESCALATE"


class Urgency(str, Enum):
    ROUTINE   = "ROUTINE"
    URGENT    = "URGENT"
    IMMEDIATE = "IMMEDIATE"


class ProcessComplianceResult(BaseModel):
    """Layer 1 — Did the associate follow the 10-step call flow?"""
    compliance_score: float           = 0.0

    introduction_done: bool           = False
    qualification_done: bool          = False
    ai_discussion_done: bool          = False
    kalvium_explained: bool           = False
    webinar_invited: bool             = False
    webinar_explained: bool           = False
    email_collected: bool             = False
    email_repeated: bool              = False
    email_confirmed: bool             = False
    timing_discussed: bool            = False
    attendance_confirmed: bool        = False

    completed_steps: list[str]        = Field(default_factory=list)
    missing_steps:   list[str]        = Field(default_factory=list)
    compliance_notes: str             = ""


class EngagementAuthenticityResult(BaseModel):
    """Layer 2 — Was the student genuinely engaged, or scripted/passive/fake?"""
    engagement_score: float           = 0.0
    natural_interaction_score: float  = 0.0

    passive_response_ratio: float     = 0.0
    one_word_reply_ratio: float       = 0.0
    meaningful_question_count: int    = 0
    contextual_response_count: int    = 0

    showed_curiosity: bool            = False
    showed_realistic_confusion: bool  = False
    asked_relevant_questions: bool    = False
    gave_personal_context: bool       = False
    natural_hesitation: bool          = False

    only_filler_responses: bool       = False
    robotic_agreement: bool           = False
    no_meaningful_participation: bool = False
    unnatural_comfort_level: bool     = False

    suspicious_patterns: list[str]    = Field(default_factory=list)
    engagement_observations: list[str] = Field(default_factory=list)


class EmailRegistrationResult(BaseModel):
    """Layer 3 — MOST IMPORTANT. Email chain + registration quality."""
    registration_authenticity_score: float = 0.0
    email_authenticity_score: float         = 0.0
    webinar_understanding_score: float      = 0.0
    fake_registration_probability: float    = 0.0

    email_collected: bool              = False
    email_repeated_by_associate: bool  = False
    email_spelling_confirmed: bool     = False
    email_confirmed_by_student: bool   = False

    webinar_explained: bool            = False
    webinar_purpose_understood: bool   = False
    webinar_timing_discussed: bool     = False
    joining_process_discussed: bool    = False

    rushed_registration: bool          = False
    associate_dominated_call: bool     = False
    student_passive_throughout: bool   = False

    student_asked_about_webinar: bool  = False
    student_confirmed_attendance: bool = False
    realistic_questions_about_joining: bool = False

    red_flags: list[str]               = Field(default_factory=list)
    genuine_signals: list[str]         = Field(default_factory=list)
    registration_notes: str            = ""


class AttendancePredictionResult(BaseModel):
    """Layer 4 — PRIMARY metric: will this student actually attend?"""
    attendance_probability: float       = 0.0
    attendance_confidence: AttendanceConfidence = AttendanceConfidence.LOW
    no_show_risk: float                 = 1.0

    email_verified: bool                = False
    timing_confirmed: bool              = False
    webinar_purpose_clear: bool         = False
    student_engaged: bool               = False
    genuine_interest_shown: bool        = False
    attendance_verbally_confirmed: bool = False
    joining_instructions_understood: bool = False

    no_webinar_understanding: bool      = False
    passive_interaction: bool           = False
    no_timing_awareness: bool           = False
    fake_sounding_registration: bool    = False
    no_joining_discussion: bool         = False

    commitment_quotes: list[str]        = Field(default_factory=list)
    risk_factors: list[str]             = Field(default_factory=list)
    prediction_rationale: str           = ""


class FraudDetectionResult(BaseModel):
    """Layer 5 — Fake/suspicious registration detection."""
    fraud_risk_level: FraudRiskLevel    = FraudRiskLevel.HIGH
    fake_probability: float             = 1.0
    booking_authenticity_score: float   = 0.0

    # Tier-1: any one → HIGH RISK
    no_student_speech: bool             = False
    associate_monologue: bool           = False
    no_email_collected: bool            = False
    ultra_short_call: bool              = False
    zero_qualification: bool            = False

    # Tier-2: strong indicators
    no_webinar_explanation: bool        = False
    no_natural_questions: bool          = False
    instant_agreement_pattern: bool     = False
    overly_casual_relationship: bool    = False
    scripted_responses: bool            = False
    no_personal_academic_context: bool  = False

    relationship_indicators: list[str]  = Field(default_factory=list)
    suspicious_behavior_flags: list[str] = Field(default_factory=list)
    red_flags: list[str]                = Field(default_factory=list)

    tier1_flag_count: int               = 0
    tier2_flag_count: int               = 0
    fraud_reasoning: str                = ""


class ManagerSummaryResult(BaseModel):
    manager_summary: str                = ""
    key_concerns: list[str]             = Field(default_factory=list)
    positive_signals: list[str]         = Field(default_factory=list)
    coaching_recommendations: list[str] = Field(default_factory=list)
    associate_risk_note: str            = ""


class KalviumAuditReport(BaseModel):
    """Complete output of the 5-layer analysis."""

    audit_id: str
    associate_id: str               = ""
    associate_name: str             = ""
    lead_id: str                    = ""
    prospect_name: str              = ""
    call_date: str                  = ""
    recording_file: str             = ""
    duration_seconds: float         = 0.0
    language_detected: str          = "hi-IN"

    talk_ratio_associate: float     = 0.0
    talk_ratio_prospect: float      = 0.0
    talk_ratio_parent: float        = 0.0
    silence_ratio: float            = 0.0

    compliance:      Optional[ProcessComplianceResult]       = None
    engagement:      Optional[EngagementAuthenticityResult]  = None
    registration:    Optional[EmailRegistrationResult]       = None
    attendance:      Optional[AttendancePredictionResult]    = None
    fraud:           Optional[FraudDetectionResult]          = None
    manager_summary: Optional[ManagerSummaryResult]          = None

    # Flattened scores for dashboard/sheets
    final_classification: FinalClassification   = FinalClassification.HIGH_RISK
    compliance_score: float                     = 0.0
    engagement_score: float                     = 0.0
    registration_authenticity_score: float      = 0.0
    attendance_probability: float               = 0.0
    attendance_confidence: str                  = "LOW"
    fake_probability: float                     = 1.0
    fraud_risk_level: FraudRiskLevel            = FraudRiskLevel.HIGH

    email_collected: bool                       = False
    email_confirmed: bool                       = False
    webinar_explained: bool                     = False
    webinar_understood: bool                    = False
    qualification_completed: bool               = False

    passive_response_ratio: float               = 0.0
    red_flags: list[str]                        = Field(default_factory=list)
    missing_steps: list[str]                    = Field(default_factory=list)
    relationship_indicators: list[str]          = Field(default_factory=list)
    behavioral_observations: list[str]          = Field(default_factory=list)
    suspicious_behavior_flags: list[str]        = Field(default_factory=list)

    recommended_action: RecommendedAction       = RecommendedAction.INVESTIGATE
    urgency: Urgency                            = Urgency.IMMEDIATE
    no_show_risk: float                         = 1.0
