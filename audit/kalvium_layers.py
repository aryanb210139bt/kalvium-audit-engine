"""
audit/kalvium_layers.py
All 5 analysis layers in one module.
Each layer is a pure function that takes a transcript and returns a Pydantic model.
Single OpenAI call per layer — parallel execution in pipeline.

PHILOSOPHY: Webinar attendance authenticity, NOT sales conversion.
"""
from __future__ import annotations
import json
import logging
import re
from typing import Any

from openai import OpenAI

from config.kalvium_models import (
    ProcessComplianceResult,
    EngagementAuthenticityResult,
    EmailRegistrationResult,
    AttendancePredictionResult,
    AttendanceConfidence,
    FraudDetectionResult,
    FraudRiskLevel,
    ManagerSummaryResult,
)
from config.models import Utterance, Speaker
from config.settings import get_settings

logger   = logging.getLogger(__name__)
settings = get_settings()

_FILLER = re.compile(
    r"^(haan|ha|ok|okay|ji|hmm|hm|theek\s*hai|thik\s*hai|acha|accha|"
    r"sure|yes|no|nahi|nope|uh|um|ya|yep|right|fine|got\s*it|"
    r"samajh\s*gaya|samajh\s*gayi|han)\.?$",
    re.IGNORECASE,
)


def _llm(prompt: str, model: str = "gpt-4o", max_tokens: int = 1200) -> dict:
    """Single GPT-4o call; returns parsed JSON or empty dict on failure."""
    client = OpenAI(api_key=settings.openai_api_key)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as exc:
        logger.warning(f"LLM call failed: {exc}")
        return {}


def _student_lines(utterances: list[Utterance]) -> str:
    lines = [
        f"[{int(u.start_time//60):02d}:{int(u.start_time%60):02d}] {u.english_text}"
        for u in utterances
        if u.speaker in (Speaker.STUDENT, Speaker.UNKNOWN)
    ]
    return "\n".join(lines[:60]) or "(no student speech detected)"


def _filler_ratio(utterances: list[Utterance]) -> float:
    student = [u for u in utterances if u.speaker in (Speaker.STUDENT, Speaker.UNKNOWN)]
    if not student:
        return 1.0
    fillers = sum(1 for u in student if _FILLER.match(u.english_text.strip()))
    return round(fillers / len(student), 3)


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 1 — PROCESS COMPLIANCE
# ═══════════════════════════════════════════════════════════════════════════════

COMPLIANCE_PROMPT = """You are an expert call quality evaluator for Kalvium's webinar registration team.

CONTEXT:
A Kalvium associate called a student (Class 12 PCM/PCMB or dropper) to:
1. Introduce themselves
2. Qualify the student (PCM/PCMB? Class 12? Dropper?)
3. Discuss AI/job market changes
4. Briefly explain Kalvium
5. Invite the student to a FREE webinar/demo
6. Explain what the webinar is about
7. Collect the student's email ID
8. Repeat/spell back the email ID
9. Confirm the student received/will receive the email
10. Discuss webinar timing/date
11. Confirm the student will attend

IMPORTANT: This is NOT a sales call. No purchase is being made.
The goal is simply to register genuine students for a free webinar.

TRANSCRIPT:
{transcript}

Evaluate each step strictly. Return ONLY valid JSON:
{{
  "compliance_score": <0-100>,
  "introduction_done": <true/false>,
  "qualification_done": <true/false>,
  "ai_discussion_done": <true/false>,
  "kalvium_explained": <true/false>,
  "webinar_invited": <true/false>,
  "webinar_explained": <true/false>,
  "email_collected": <true/false>,
  "email_repeated": <true/false>,
  "email_confirmed": <true/false>,
  "timing_discussed": <true/false>,
  "attendance_confirmed": <true/false>,
  "completed_steps": ["list of steps that were clearly done"],
  "missing_steps": ["list of steps that were skipped or incomplete"],
  "compliance_notes": "brief note on biggest gaps"
}}"""


def analyze_compliance(
    utterances: list[Utterance], full_transcript: str
) -> ProcessComplianceResult:
    prompt = COMPLIANCE_PROMPT.format(transcript=full_transcript[:7000])
    data   = _llm(prompt, max_tokens=800)

    if not data:
        return ProcessComplianceResult()

    # Rule-based floor check — regex fast pass
    transcript_lower = full_transcript.lower()
    email_mentioned  = bool(re.search(r"@|email|gmail|mail\s*id", transcript_lower))
    email_repeated   = bool(re.search(
        r"(repeat|spell|double|again|bola|bole|phir\s*se|dobara)", transcript_lower
    ))

    return ProcessComplianceResult(
        compliance_score      = float(data.get("compliance_score", 0)),
        introduction_done     = bool(data.get("introduction_done", False)),
        qualification_done    = bool(data.get("qualification_done", False)),
        ai_discussion_done    = bool(data.get("ai_discussion_done", False)),
        kalvium_explained     = bool(data.get("kalvium_explained", False)),
        webinar_invited       = bool(data.get("webinar_invited", False)),
        webinar_explained     = bool(data.get("webinar_explained", False)),
        email_collected       = bool(data.get("email_collected", email_mentioned)),
        email_repeated        = bool(data.get("email_repeated", email_repeated)),
        email_confirmed       = bool(data.get("email_confirmed", False)),
        timing_discussed      = bool(data.get("timing_discussed", False)),
        attendance_confirmed  = bool(data.get("attendance_confirmed", False)),
        completed_steps       = data.get("completed_steps", []),
        missing_steps         = data.get("missing_steps", []),
        compliance_notes      = data.get("compliance_notes", ""),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 2 — ENGAGEMENT AUTHENTICITY
# ═══════════════════════════════════════════════════════════════════════════════

ENGAGEMENT_PROMPT = """You are a behavioral authenticity analyst specializing in Indian ed-tech call centers.

CONTEXT:
A Kalvium associate is inviting a student to a FREE webinar. No purchase involved.
Your job is to detect whether the student's participation is GENUINE or FAKE/SCRIPTED.

GENUINE engagement looks like:
- Student asks natural questions ("webinar kab hai?", "joining kaise karein?", "kya padhate hain?")
- Student mentions their stream/marks/college plans naturally
- Student shows some hesitation or asks to clarify
- Student reacts to information (AI job market, Kalvium program)
- Student says something personal or contextual
- Natural pauses, confusion, or follow-up questions

FAKE/SUSPICIOUS engagement looks like:
- Only "haan", "ok", "theek hai", "yes" responses
- No real questions about the webinar
- Instant agreement to everything without curiosity
- Scripted-sounding responses
- No personal academic context shared
- Unnatural comfort level (sounds like they know the associate)
- Call feels coached or pre-arranged
- Associate speaks almost the entire call, student just confirms

STUDENT UTTERANCES ONLY:
{student_lines}

FULL TRANSCRIPT (for context):
{transcript}

Analyze the STUDENT's behavior only. Return ONLY valid JSON:
{{
  "engagement_score": <0-100>,
  "natural_interaction_score": <0-100>,
  "passive_response_ratio": <0.0-1.0>,
  "one_word_reply_ratio": <0.0-1.0>,
  "meaningful_question_count": <integer>,
  "contextual_response_count": <integer>,
  "showed_curiosity": <true/false>,
  "showed_realistic_confusion": <true/false>,
  "asked_relevant_questions": <true/false>,
  "gave_personal_context": <true/false>,
  "natural_hesitation": <true/false>,
  "only_filler_responses": <true/false>,
  "robotic_agreement": <true/false>,
  "no_meaningful_participation": <true/false>,
  "unnatural_comfort_level": <true/false>,
  "suspicious_patterns": ["list specific suspicious patterns observed"],
  "engagement_observations": ["list of specific genuine engagement signals"]
}}"""


def analyze_engagement(
    utterances: list[Utterance], full_transcript: str, talk_ratio_associate: float
) -> EngagementAuthenticityResult:
    student_text = _student_lines(utterances)
    prompt       = ENGAGEMENT_PROMPT.format(
        student_lines=student_text,
        transcript=full_transcript[:6000],
    )
    data = _llm(prompt, max_tokens=900)

    filler_r = _filler_ratio(utterances)

    if not data:
        return EngagementAuthenticityResult(
            passive_response_ratio=filler_r,
            no_meaningful_participation=True,
        )

    # Override with measured filler ratio if LLM underestimates
    passive_r = max(float(data.get("passive_response_ratio", filler_r)), filler_r * 0.8)

    return EngagementAuthenticityResult(
        engagement_score          = float(data.get("engagement_score", 0)),
        natural_interaction_score = float(data.get("natural_interaction_score", 0)),
        passive_response_ratio    = round(passive_r, 3),
        one_word_reply_ratio      = float(data.get("one_word_reply_ratio", filler_r)),
        meaningful_question_count = int(data.get("meaningful_question_count", 0)),
        contextual_response_count = int(data.get("contextual_response_count", 0)),
        showed_curiosity          = bool(data.get("showed_curiosity", False)),
        showed_realistic_confusion= bool(data.get("showed_realistic_confusion", False)),
        asked_relevant_questions  = bool(data.get("asked_relevant_questions", False)),
        gave_personal_context     = bool(data.get("gave_personal_context", False)),
        natural_hesitation        = bool(data.get("natural_hesitation", False)),
        only_filler_responses     = bool(data.get("only_filler_responses", filler_r > 0.85)),
        robotic_agreement         = bool(data.get("robotic_agreement", False)),
        no_meaningful_participation = bool(data.get("no_meaningful_participation", False)),
        unnatural_comfort_level   = bool(data.get("unnatural_comfort_level", False)),
        suspicious_patterns       = data.get("suspicious_patterns", []),
        engagement_observations   = data.get("engagement_observations", []),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 3 — EMAIL & REGISTRATION AUTHENTICITY
# ═══════════════════════════════════════════════════════════════════════════════

REGISTRATION_PROMPT = """You are an expert in detecting fake webinar registrations.

CONTEXT:
Kalvium associates collect email IDs and register students for a FREE webinar.
Some associates create FAKE registrations (friends, family, fake leads) to hit targets.

The EMAIL VERIFICATION CHAIN is the single most important authenticity signal:
  Step A: Associate asks for email
  Step B: Student gives email
  Step C: Associate REPEATS email back / spells it out
  Step D: Student CONFIRMS it's correct
  Step E: Associate mentions sending invite/joining link
  Step F: Student acknowledges

A GENUINE registration always has steps A-D at minimum.
A FAKE registration often rushes through or skips C-F.

WEBINAR UNDERSTANDING is the second most important signal:
- Did the student understand what they're registering for?
- Did they know when the webinar is?
- Did they ask about joining?

STRONG GENUINE SIGNALS:
- Complete email verification chain
- Student asked clarifying questions about webinar
- Student mentioned schedule conflict or confirmed timing
- Natural back-and-forth about the webinar content
- Student's questions sounded like a real Class 12 student

STRONG SUSPICIOUS SIGNALS:
- Email collected but never verified/repeated
- Student agreed immediately with no questions
- No mention of when/how to join the webinar
- Call was very rushed
- Associate spoke >80% of the time
- No natural conversation about what the webinar is

TRANSCRIPT:
{transcript}

Return ONLY valid JSON:
{{
  "registration_authenticity_score": <0-100>,
  "email_authenticity_score": <0-100>,
  "webinar_understanding_score": <0-100>,
  "fake_registration_probability": <0.0-1.0>,
  "email_collected": <true/false>,
  "email_repeated_by_associate": <true/false>,
  "email_spelling_confirmed": <true/false>,
  "email_confirmed_by_student": <true/false>,
  "webinar_explained": <true/false>,
  "webinar_purpose_understood": <true/false>,
  "webinar_timing_discussed": <true/false>,
  "joining_process_discussed": <true/false>,
  "rushed_registration": <true/false>,
  "associate_dominated_call": <true/false>,
  "student_passive_throughout": <true/false>,
  "student_asked_about_webinar": <true/false>,
  "student_confirmed_attendance": <true/false>,
  "realistic_questions_about_joining": <true/false>,
  "red_flags": ["specific red flags found"],
  "genuine_signals": ["specific genuine signals found"],
  "registration_notes": "brief analysis summary"
}}"""


def analyze_registration(
    utterances: list[Utterance], full_transcript: str, duration_sec: float,
    talk_ratio_associate: float
) -> EmailRegistrationResult:
    prompt = REGISTRATION_PROMPT.format(transcript=full_transcript[:7000])
    data   = _llm(prompt, max_tokens=1000)

    # Rule-based signals
    tl = full_transcript.lower()
    email_re      = bool(re.search(r"@|email|gmail|yahoo|mail", tl))
    timing_re     = bool(re.search(r"(date|time|saturday|sunday|baje|bajkar|pm|am|schedule|timing)", tl))
    rushed        = duration_sec < 180  # under 3 minutes
    assoc_dom     = talk_ratio_associate > 80

    if not data:
        return EmailRegistrationResult(
            email_collected          = email_re,
            webinar_timing_discussed = timing_re,
            rushed_registration      = rushed,
            associate_dominated_call = assoc_dom,
            fake_registration_probability = 0.8 if rushed else 0.5,
        )

    return EmailRegistrationResult(
        registration_authenticity_score  = float(data.get("registration_authenticity_score", 0)),
        email_authenticity_score         = float(data.get("email_authenticity_score", 0)),
        webinar_understanding_score      = float(data.get("webinar_understanding_score", 0)),
        fake_registration_probability    = float(data.get("fake_registration_probability", 0.5)),
        email_collected                  = bool(data.get("email_collected", email_re)),
        email_repeated_by_associate      = bool(data.get("email_repeated_by_associate", False)),
        email_spelling_confirmed         = bool(data.get("email_spelling_confirmed", False)),
        email_confirmed_by_student       = bool(data.get("email_confirmed_by_student", False)),
        webinar_explained                = bool(data.get("webinar_explained", False)),
        webinar_purpose_understood       = bool(data.get("webinar_purpose_understood", False)),
        webinar_timing_discussed         = bool(data.get("webinar_timing_discussed", timing_re)),
        joining_process_discussed        = bool(data.get("joining_process_discussed", False)),
        rushed_registration              = bool(data.get("rushed_registration", rushed)),
        associate_dominated_call         = bool(data.get("associate_dominated_call", assoc_dom)),
        student_passive_throughout       = bool(data.get("student_passive_throughout", False)),
        student_asked_about_webinar      = bool(data.get("student_asked_about_webinar", False)),
        student_confirmed_attendance     = bool(data.get("student_confirmed_attendance", False)),
        realistic_questions_about_joining= bool(data.get("realistic_questions_about_joining", False)),
        red_flags                        = data.get("red_flags", []),
        genuine_signals                  = data.get("genuine_signals", []),
        registration_notes               = data.get("registration_notes", ""),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 4 — ATTENDANCE PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════

ATTENDANCE_PROMPT = """You are an attendance prediction specialist for online webinars in India.

CONTEXT:
Kalvium runs free webinars for Class 12 PCM/PCMB students and droppers.
You must predict the probability that THIS specific student will actually attend.

The PRIMARY predictor of genuine attendance is:
1. Complete email verification (student confirmed email)
2. Student understood what the webinar is and when
3. Student showed genuine interest in the topic
4. Natural, engaged conversation (not passive/fake)
5. Timing was confirmed and student acknowledged it

Strong NO-SHOW predictors:
1. Student never confirmed the email
2. Student had no idea what the webinar was about
3. Student was completely passive (just "haan ok" throughout)
4. Registration was rushed (<3 minutes)
5. Call felt fake/coached/pre-arranged
6. Student showed no curiosity about the content or joining process

Indian webinar context:
- Class 12 students are busy, so timing confirmation matters a lot
- Students who ask "joining kaise karna hai?" are more likely to attend
- Students who understood the AI/career relevance attend at higher rates
- Students who just gave email without any conversation rarely show up

TRANSCRIPT:
{transcript}

COMPLIANCE SIGNALS:
- Email confirmed: {email_confirmed}
- Timing discussed: {timing_discussed}
- Webinar explained: {webinar_explained}
- Student engagement level: {engagement_score}/100
- Fake registration probability: {fake_probability}

Return ONLY valid JSON:
{{
  "attendance_probability": <0.0-1.0>,
  "attendance_confidence": "HIGH/MEDIUM/LOW",
  "no_show_risk": <0.0-1.0>,
  "email_verified": <true/false>,
  "timing_confirmed": <true/false>,
  "webinar_purpose_clear": <true/false>,
  "student_engaged": <true/false>,
  "genuine_interest_shown": <true/false>,
  "attendance_verbally_confirmed": <true/false>,
  "joining_instructions_understood": <true/false>,
  "no_webinar_understanding": <true/false>,
  "passive_interaction": <true/false>,
  "no_timing_awareness": <true/false>,
  "fake_sounding_registration": <true/false>,
  "no_joining_discussion": <true/false>,
  "commitment_quotes": ["exact phrases student said showing intent to attend"],
  "risk_factors": ["specific reasons this student might not attend"],
  "prediction_rationale": "2-3 sentence explanation of the probability estimate"
}}"""


def analyze_attendance(
    utterances: list[Utterance],
    full_transcript: str,
    compliance: ProcessComplianceResult,
    engagement: EngagementAuthenticityResult,
    registration: EmailRegistrationResult,
) -> AttendancePredictionResult:

    # Short-circuit: if registration is clearly fake, attendance is near-zero
    if registration.fake_registration_probability > 0.80:
        return AttendancePredictionResult(
            attendance_probability   = 0.03,
            attendance_confidence    = AttendanceConfidence.HIGH,
            no_show_risk             = 0.97,
            fake_sounding_registration = True,
            no_webinar_understanding = not registration.webinar_purpose_understood,
            passive_interaction      = engagement.no_meaningful_participation,
            prediction_rationale     = "Near-zero attendance predicted: registration flagged as very likely fake.",
        )

    prompt = ATTENDANCE_PROMPT.format(
        transcript       = full_transcript[:6000],
        email_confirmed  = compliance.email_confirmed,
        timing_discussed = compliance.timing_discussed,
        webinar_explained= compliance.webinar_explained,
        engagement_score = engagement.engagement_score,
        fake_probability = registration.fake_registration_probability,
    )
    data = _llm(prompt, max_tokens=900)

    if not data:
        # Rule-based fallback
        prob = 0.5
        if compliance.email_confirmed:        prob += 0.15
        if compliance.timing_discussed:       prob += 0.10
        if compliance.webinar_explained:      prob += 0.10
        if engagement.engagement_score > 60:  prob += 0.10
        if registration.fake_registration_probability > 0.5: prob -= 0.30
        return AttendancePredictionResult(
            attendance_probability = round(min(max(prob, 0), 1), 2),
            no_show_risk           = round(1 - min(max(prob, 0), 1), 2),
        )

    prob = float(data.get("attendance_probability", 0.3))
    conf_raw = data.get("attendance_confidence", "LOW")
    try:
        conf = AttendanceConfidence(conf_raw)
    except ValueError:
        conf = AttendanceConfidence.LOW

    return AttendancePredictionResult(
        attendance_probability          = round(prob, 3),
        attendance_confidence           = conf,
        no_show_risk                    = round(1 - prob, 3),
        email_verified                  = bool(data.get("email_verified", False)),
        timing_confirmed                = bool(data.get("timing_confirmed", False)),
        webinar_purpose_clear           = bool(data.get("webinar_purpose_clear", False)),
        student_engaged                 = bool(data.get("student_engaged", False)),
        genuine_interest_shown          = bool(data.get("genuine_interest_shown", False)),
        attendance_verbally_confirmed   = bool(data.get("attendance_verbally_confirmed", False)),
        joining_instructions_understood = bool(data.get("joining_instructions_understood", False)),
        no_webinar_understanding        = bool(data.get("no_webinar_understanding", False)),
        passive_interaction             = bool(data.get("passive_interaction", False)),
        no_timing_awareness             = bool(data.get("no_timing_awareness", False)),
        fake_sounding_registration      = bool(data.get("fake_sounding_registration", False)),
        no_joining_discussion           = bool(data.get("no_joining_discussion", False)),
        commitment_quotes               = data.get("commitment_quotes", []),
        risk_factors                    = data.get("risk_factors", []),
        prediction_rationale            = data.get("prediction_rationale", ""),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 5 — FRAUD DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

FRAUD_PROMPT = """You are a fraud detection specialist for an Indian ed-tech company's webinar registration system.

CONTEXT:
Some associates register fake/suspicious leads to hit booking targets. These include:
- Registering friends or family members
- Calling people who never agreed to attend
- Coaching people to sound genuine
- Using fake email IDs
- Running rushed, hollow registrations

TIER-1 HARD DISQUALIFIERS (any single one = HIGH RISK):
1. Student speaks less than 5% of the call (associate monologue)
2. No email was collected at all
3. Call was under 90 seconds
4. Zero evidence of student qualification (PCM/PCMB check)
5. Student speech detected as zero / completely absent

TIER-2 STRONG FRAUD INDICATORS (multiple = HIGH/MEDIUM risk):
1. No explanation of what the webinar is about
2. Student asked no questions of any kind
3. Instant agreement to everything with no hesitation
4. Overly casual/familiar tone (bro, yaar, excessive familiarity)
5. Scripted-sounding responses (too perfect, too fast)
6. No mention of the student's class/stream/college
7. Webinar timing never discussed
8. No natural confusion or need for clarification
9. Email given but never verified/spelled back

RELATIONSHIP FRAUD INDICATORS:
- Uses names/nicknames casually (sounds like they know each other)
- Laughing or overly casual banter not typical of a cold call
- Student answers without hesitation on sensitive questions (stream, marks)
- Associate skips qualification as if they already know the student

GENUINE REGISTRATION INDICATORS:
- Student asked at least one real question
- Student mentioned their stream or academic context
- Email was properly verified
- Webinar timing was discussed
- Natural pause/hesitation/clarification happened
- Student showed at least some confusion or curiosity

TRANSCRIPT:
{transcript}

CONTEXT SIGNALS:
- Duration: {duration}s
- Associate talk ratio: {associate_pct}%
- Student talk ratio: {student_pct}%
- Email chain completed: {email_chain}
- Compliance score: {compliance}/100

Evaluate fraud indicators carefully. Return ONLY valid JSON:
{{
  "fraud_risk_level": "LOW/MEDIUM/HIGH",
  "fake_probability": <0.0-1.0>,
  "booking_authenticity_score": <0-100>,
  "no_student_speech": <true/false>,
  "associate_monologue": <true/false>,
  "no_email_collected": <true/false>,
  "ultra_short_call": <true/false>,
  "zero_qualification": <true/false>,
  "no_webinar_explanation": <true/false>,
  "no_natural_questions": <true/false>,
  "instant_agreement_pattern": <true/false>,
  "overly_casual_relationship": <true/false>,
  "scripted_responses": <true/false>,
  "no_personal_academic_context": <true/false>,
  "relationship_indicators": ["list specific relationship fraud signals"],
  "suspicious_behavior_flags": ["list specific suspicious behavioral patterns"],
  "red_flags": ["all red flags found, most serious first"],
  "tier1_flag_count": <integer>,
  "tier2_flag_count": <integer>,
  "fraud_reasoning": "2-3 sentence explanation of the fraud risk level"
}}"""


def analyze_fraud(
    utterances: list[Utterance],
    full_transcript: str,
    duration_sec: float,
    talk_ratio_associate: float,
    talk_ratio_student: float,
    compliance: ProcessComplianceResult,
    registration: EmailRegistrationResult,
) -> FraudDetectionResult:

    # Deterministic tier-1 checks (no LLM needed)
    student_utts   = [u for u in utterances if u.speaker in (Speaker.STUDENT, Speaker.UNKNOWN)]
    no_student     = len(student_utts) == 0 or talk_ratio_student < 3
    monologue      = talk_ratio_associate > 90
    no_email       = not compliance.email_collected
    ultra_short    = duration_sec < 90
    zero_qual      = not compliance.qualification_done

    tier1 = sum([no_student, monologue, no_email, ultra_short, zero_qual])

    if tier1 >= 1:
        # Skip LLM — hard rule failure
        flags = []
        if no_student:   flags.append("No student speech detected — monologue or ghost registration")
        if monologue:    flags.append(f"Associate spoke {talk_ratio_associate:.0f}% — student silent")
        if no_email:     flags.append("Email never collected — fundamental registration failure")
        if ultra_short:  flags.append(f"Call only {duration_sec:.0f}s — too short for genuine registration")
        if zero_qual:    flags.append("No student qualification done — stream/class never confirmed")
        return FraudDetectionResult(
            fraud_risk_level       = FraudRiskLevel.HIGH,
            fake_probability       = 0.95,
            booking_authenticity_score = 5.0,
            no_student_speech      = no_student,
            associate_monologue    = monologue,
            no_email_collected     = no_email,
            ultra_short_call       = ultra_short,
            zero_qualification     = zero_qual,
            tier1_flag_count       = tier1,
            red_flags              = flags,
            fraud_reasoning        = f"{tier1} Tier-1 disqualifier(s) detected. Automatic HIGH RISK.",
        )

    email_chain = (
        compliance.email_collected and
        compliance.email_repeated and
        compliance.email_confirmed
    )

    prompt = FRAUD_PROMPT.format(
        transcript     = full_transcript[:7000],
        duration       = int(duration_sec),
        associate_pct  = int(talk_ratio_associate),
        student_pct    = int(talk_ratio_student),
        email_chain    = email_chain,
        compliance     = int(compliance.compliance_score),
    )
    data = _llm(prompt, max_tokens=1000)

    if not data:
        fp = 0.3
        if not email_chain:                  fp += 0.20
        if registration.rushed_registration: fp += 0.15
        return FraudDetectionResult(
            fake_probability       = round(fp, 2),
            fraud_risk_level       = FraudRiskLevel.HIGH if fp > 0.6 else FraudRiskLevel.MEDIUM,
            no_student_speech      = no_student,
            associate_monologue    = monologue,
            no_email_collected     = no_email,
            ultra_short_call       = ultra_short,
            zero_qualification     = zero_qual,
            tier1_flag_count       = tier1,
        )

    risk_raw = data.get("fraud_risk_level", "HIGH")
    try:
        risk = FraudRiskLevel(risk_raw)
    except ValueError:
        risk = FraudRiskLevel.HIGH

    return FraudDetectionResult(
        fraud_risk_level             = risk,
        fake_probability             = float(data.get("fake_probability", 0.5)),
        booking_authenticity_score   = float(data.get("booking_authenticity_score", 0)),
        no_student_speech            = no_student,
        associate_monologue          = monologue,
        no_email_collected           = no_email,
        ultra_short_call             = ultra_short,
        zero_qualification           = zero_qual,
        no_webinar_explanation       = bool(data.get("no_webinar_explanation", False)),
        no_natural_questions         = bool(data.get("no_natural_questions", False)),
        instant_agreement_pattern    = bool(data.get("instant_agreement_pattern", False)),
        overly_casual_relationship   = bool(data.get("overly_casual_relationship", False)),
        scripted_responses           = bool(data.get("scripted_responses", False)),
        no_personal_academic_context = bool(data.get("no_personal_academic_context", False)),
        relationship_indicators      = data.get("relationship_indicators", []),
        suspicious_behavior_flags    = data.get("suspicious_behavior_flags", []),
        red_flags                    = data.get("red_flags", []),
        tier1_flag_count             = tier1,
        tier2_flag_count             = int(data.get("tier2_flag_count", 0)),
        fraud_reasoning              = data.get("fraud_reasoning", ""),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MANAGER SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

SUMMARY_PROMPT = """You are a sales manager at Kalvium reviewing a webinar registration call.

The AI has completed 5-layer analysis. Summarize for the manager in plain English.

ANALYSIS RESULTS:
- Final Classification: {classification}
- Compliance Score: {compliance}/100
- Engagement Score: {engagement}/100
- Registration Authenticity: {registration}/100
- Attendance Probability: {attendance}%
- Fake Probability: {fake}%
- Fraud Risk: {risk}
- Red Flags: {red_flags}
- Missing Steps: {missing_steps}

KEY QUESTIONS TO ADDRESS:
1. Is this registration genuine?
2. Will this student likely attend the webinar?
3. What should the manager do (approve/investigate/escalate)?
4. What coaching does the associate need?

Write for a non-technical manager. Be direct. Return ONLY valid JSON:
{{
  "manager_summary": "3-4 sentence plain English summary for the manager",
  "key_concerns": ["top 3 concerns in order of importance"],
  "positive_signals": ["things that went well or genuine signals found"],
  "coaching_recommendations": ["specific coaching points for the associate"],
  "associate_risk_note": "one sentence on associate risk level"
}}"""


def generate_manager_summary(
    classification: str,
    compliance_score: float,
    engagement_score: float,
    registration_score: float,
    attendance_prob: float,
    fake_prob: float,
    fraud_risk: str,
    red_flags: list[str],
    missing_steps: list[str],
) -> ManagerSummaryResult:
    prompt = SUMMARY_PROMPT.format(
        classification = classification,
        compliance     = int(compliance_score),
        engagement     = int(engagement_score),
        registration   = int(registration_score),
        attendance     = int(attendance_prob * 100),
        fake           = int(fake_prob * 100),
        risk           = fraud_risk,
        red_flags      = "; ".join(red_flags[:5]) or "None",
        missing_steps  = "; ".join(missing_steps[:5]) or "None",
    )
    data = _llm(prompt, max_tokens=700)

    if not data:
        return ManagerSummaryResult(
            manager_summary=f"Classification: {classification}. Attendance probability: {int(attendance_prob*100)}%. Fake probability: {int(fake_prob*100)}%.",
        )

    return ManagerSummaryResult(
        manager_summary          = data.get("manager_summary", ""),
        key_concerns             = data.get("key_concerns", []),
        positive_signals         = data.get("positive_signals", []),
        coaching_recommendations = data.get("coaching_recommendations", []),
        associate_risk_note      = data.get("associate_risk_note", ""),
    )
