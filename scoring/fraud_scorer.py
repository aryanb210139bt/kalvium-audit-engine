"""
scoring/fraud_scorer.py
Aggregate all 5 layer results into final classification.
"""
from __future__ import annotations
import logging

from config.kalvium_models import (
    KalviumAuditReport, FinalClassification, FraudRiskLevel,
    RecommendedAction, Urgency,
)
from audit.kalvium_layers import generate_manager_summary

logger = logging.getLogger(__name__)


class FraudScorer:

    def score(self, report: KalviumAuditReport) -> KalviumAuditReport:
        c   = report.compliance
        eng = report.engagement
        reg = report.registration
        att = report.attendance
        frd = report.fraud

        if not all([c, eng, reg, att, frd]):
            logger.error("FraudScorer: missing layer results")
            return report

        composite = (
            c.compliance_score                     * 0.15 +
            eng.engagement_score                   * 0.20 +
            reg.registration_authenticity_score    * 0.35 +
            att.attendance_probability * 100       * 0.20 +
            frd.booking_authenticity_score         * 0.10
        )

        fake_prob  = _blend_fake_prob(frd, reg, eng, att)
        fraud_risk = _fraud_risk(fake_prob, frd.tier1_flag_count, frd.tier2_flag_count)
        fc         = _classify(fake_prob, composite, frd.tier1_flag_count)
        action     = _action(fc, fraud_risk)
        urgency    = _urgency(fc, frd.tier1_flag_count)

        all_flags = list(dict.fromkeys(
            frd.red_flags + reg.red_flags + eng.suspicious_patterns
        ))[:10]

        ms = generate_manager_summary(
            classification     = fc.value,
            compliance_score   = c.compliance_score,
            engagement_score   = eng.engagement_score,
            registration_score = reg.registration_authenticity_score,
            attendance_prob    = att.attendance_probability,
            fake_prob          = fake_prob,
            fraud_risk         = fraud_risk.value,
            red_flags          = all_flags,
            missing_steps      = c.missing_steps,
        )

        report.manager_summary                 = ms
        report.final_classification            = fc
        report.compliance_score                = round(c.compliance_score, 1)
        report.engagement_score                = round(eng.engagement_score, 1)
        report.registration_authenticity_score = round(reg.registration_authenticity_score, 1)
        report.attendance_probability          = round(att.attendance_probability, 3)
        report.attendance_confidence           = att.attendance_confidence.value
        report.fake_probability                = round(fake_prob, 3)
        report.fraud_risk_level                = fraud_risk
        report.recommended_action              = action
        report.urgency                         = urgency
        report.no_show_risk                    = round(att.no_show_risk, 3)
        report.email_collected                 = c.email_collected
        report.email_confirmed                 = c.email_confirmed
        report.webinar_explained               = c.webinar_explained
        report.webinar_understood              = reg.webinar_purpose_understood
        report.qualification_completed         = c.qualification_done
        report.passive_response_ratio          = eng.passive_response_ratio
        report.red_flags                       = all_flags
        report.missing_steps                   = c.missing_steps
        report.relationship_indicators         = frd.relationship_indicators
        report.behavioral_observations         = eng.engagement_observations
        report.suspicious_behavior_flags       = frd.suspicious_behavior_flags

        logger.info(
            f"Scored: {fc.value} | fake={fake_prob:.0%} | "
            f"attend={att.attendance_probability:.0%} | {action.value}"
        )
        return report


def _blend_fake_prob(frd, reg, eng, att) -> float:
    base = (
        frd.fake_probability               * 0.45 +
        reg.fake_registration_probability  * 0.35 +
        (1 - att.attendance_probability)   * 0.10 +
        eng.passive_response_ratio         * 0.10
    )
    if frd.tier1_flag_count >= 1:
        base = max(base, 0.80)
    if frd.tier1_flag_count >= 2:
        base = max(base, 0.92)
    return round(min(base, 0.99), 3)


def _classify(fake_prob: float, composite: float, tier1: int) -> FinalClassification:
    if tier1 > 0 or fake_prob >= 0.65 or composite <= 30:
        return FinalClassification.HIGH_RISK
    if fake_prob >= 0.45 or composite <= 45:
        return FinalClassification.SUSPICIOUS
    if fake_prob >= 0.30 or composite <= 60:
        return FinalClassification.WEAK
    if fake_prob >= 0.15 or composite <= 75:
        return FinalClassification.LIKELY_GENUINE
    return FinalClassification.GENUINE


def _fraud_risk(fake_prob: float, tier1: int, tier2: int) -> FraudRiskLevel:
    if tier1 > 0 or fake_prob > 0.65:
        return FraudRiskLevel.HIGH
    if tier2 >= 3 or fake_prob > 0.35:
        return FraudRiskLevel.MEDIUM
    return FraudRiskLevel.LOW


def _action(fc: FinalClassification, risk: FraudRiskLevel) -> RecommendedAction:
    if fc == FinalClassification.HIGH_RISK:
        return RecommendedAction.ESCALATE
    if fc == FinalClassification.SUSPICIOUS:
        return RecommendedAction.INVESTIGATE
    if fc == FinalClassification.WEAK:
        return RecommendedAction.FLAG_FOR_REVIEW
    if fc == FinalClassification.LIKELY_GENUINE and risk == FraudRiskLevel.LOW:
        return RecommendedAction.APPROVE
    return RecommendedAction.MONITOR


def _urgency(fc: FinalClassification, tier1: int) -> Urgency:
    if fc == FinalClassification.HIGH_RISK or tier1 > 0:
        return Urgency.IMMEDIATE
    if fc == FinalClassification.SUSPICIOUS:
        return Urgency.URGENT
    return Urgency.ROUTINE
