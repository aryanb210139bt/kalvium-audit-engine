"""
reports/pdf_generator.py
Clean, professional white PDF report for Kalvium Demo Audit.

Layout  (~8-10 pages):
  Page 1  — Cover: score ring, session meta, key findings, quick stats
  Page 2  — Category Scores (10 categories, 2-column compact bars)
  Page 3  — Coaching Playbook (strengths + improvements + per-category tips)
  Page 4  — Sentiment & Objections
  Page 5+ — Transcript (English only, compact)
"""
from __future__ import annotations
import io
import logging
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether,
)
from reportlab.platypus.flowables import Flowable

logger = logging.getLogger(__name__)

# ── Colour palette (clean white theme) ────────────────────────────────────────
WHITE      = colors.white
PAGE_BG    = colors.white
INK        = colors.HexColor("#111827")   # near-black text
INK_DIM    = colors.HexColor("#6b7280")   # muted labels
BORDER     = colors.HexColor("#e5e7eb")   # card borders
RED        = colors.HexColor("#dc2626")   # Kalvium red
RED_LIGHT  = colors.HexColor("#fef2f2")   # red tint bg
GREEN      = colors.HexColor("#16a34a")
AMBER      = colors.HexColor("#d97706")
BLUE       = colors.HexColor("#2563eb")
BLUE_LIGHT = colors.HexColor("#eff6ff")
GRAY_BG    = colors.HexColor("#f9fafb")
STRIP_ALT  = colors.HexColor("#f3f4f6")

PAGE_W, PAGE_H = A4


def _score_color(s: float):
    if s >= 7: return GREEN
    if s >= 5: return AMBER
    return RED

def _grade_color(g: str):
    g = g.replace("+", "")
    if g == "A": return GREEN
    if g == "B": return colors.HexColor("#65a30d")
    if g == "C": return AMBER
    if g == "D": return colors.HexColor("#ea580c")
    return RED


# ── Score ring ────────────────────────────────────────────────────────────────
class ScoreRing(Flowable):
    def __init__(self, score: float, grade: str, size: float = 90):
        super().__init__()
        self.score = score; self.grade = grade; self.size = size
        self.width = size; self.height = size

    def draw(self):
        import math
        cx = cy = self.size / 2
        r  = self.size * 0.38
        lw = self.size * 0.09
        # Track
        self.canv.setStrokeColor(BORDER)
        self.canv.setLineWidth(lw)
        self.canv.circle(cx, cy, r, stroke=1, fill=0)
        # Arc
        color = _grade_color(self.grade)
        self.canv.setStrokeColor(color)
        pct   = self.score / 100
        sweep = pct * 360
        steps = max(1, int(sweep / 3))
        for i in range(steps):
            a1 = math.radians(90 - i * (sweep / steps))
            a2 = math.radians(90 - (i + 1) * (sweep / steps))
            self.canv.line(cx + r*math.cos(a1), cy + r*math.sin(a1),
                           cx + r*math.cos(a2), cy + r*math.sin(a2))
        # Score
        self.canv.setFillColor(INK)
        self.canv.setFont("Helvetica-Bold", self.size * 0.19)
        self.canv.drawCentredString(cx, cy + self.size * 0.04, str(int(self.score)))
        # Grade
        self.canv.setFillColor(color)
        self.canv.setFont("Helvetica-Bold", self.size * 0.13)
        self.canv.drawCentredString(cx, cy - self.size * 0.13, self.grade)
        # /100
        self.canv.setFillColor(INK_DIM)
        self.canv.setFont("Helvetica", self.size * 0.09)
        self.canv.drawCentredString(cx, cy - self.size * 0.25, "/100")


# ── Compact score bar ─────────────────────────────────────────────────────────
class HBar(Flowable):
    def __init__(self, label: str, score: float, weight: int = 0, bar_w: float = 95*mm):
        super().__init__()
        self.label  = label
        self.score  = score
        self.weight = weight
        self.bar_w  = bar_w
        self.height = 10
        self.width  = bar_w + 70
        self._height = 14

    def draw(self):
        bh = 7
        # Label
        self.canv.setFont("Helvetica", 7.5)
        self.canv.setFillColor(INK)
        self.canv.drawString(0, bh * 0.2, self.label[:32])
        # Weight badge
        if self.weight:
            self.canv.setFont("Helvetica", 6.5)
            self.canv.setFillColor(INK_DIM)
            self.canv.drawString(0, -4, f"{self.weight}%")
        # Track
        bx = 110
        self.canv.setFillColor(STRIP_ALT)
        self.canv.roundRect(bx, 0, self.bar_w, bh, 2, fill=1, stroke=0)
        # Fill
        fill = (self.score / 10) * self.bar_w
        self.canv.setFillColor(_score_color(self.score))
        self.canv.roundRect(bx, 0, max(2, fill), bh, 2, fill=1, stroke=0)
        # Score text
        self.canv.setFont("Helvetica-Bold", 7.5)
        self.canv.setFillColor(INK)
        self.canv.drawString(bx + self.bar_w + 5, bh * 0.2, f"{self.score:.1f}/10")


# ── Style sheet ───────────────────────────────────────────────────────────────
def _S():
    return {
        "cover_brand": ParagraphStyle("cb", fontName="Helvetica", fontSize=8,
                        textColor=RED, spaceAfter=2, letterSpacing=1.5),
        "cover_title": ParagraphStyle("ct", fontName="Helvetica-Bold", fontSize=22,
                        textColor=INK, spaceAfter=4, leading=26),
        "h2":  ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=12,
                textColor=INK, spaceAfter=3, spaceBefore=6),
        "h3":  ParagraphStyle("h3", fontName="Helvetica-Bold", fontSize=9.5,
                textColor=INK, spaceAfter=2),
        "body": ParagraphStyle("body", fontName="Helvetica", fontSize=8.5,
                 textColor=INK, spaceAfter=2, leading=12),
        "dim":  ParagraphStyle("dim", fontName="Helvetica", fontSize=7.5,
                 textColor=INK_DIM, spaceAfter=1),
        "mono": ParagraphStyle("mono", fontName="Courier", fontSize=7.5,
                 textColor=INK, spaceAfter=1, leading=11),
        "green": ParagraphStyle("gn", fontName="Helvetica", fontSize=8.5,
                  textColor=GREEN, spaceAfter=2, leading=12),
        "red":   ParagraphStyle("rd", fontName="Helvetica", fontSize=8.5,
                  textColor=RED, spaceAfter=2, leading=12),
        "amber": ParagraphStyle("am", fontName="Helvetica", fontSize=8.5,
                  textColor=AMBER, spaceAfter=2, leading=12),
        "coaching": ParagraphStyle("cch", fontName="Helvetica", fontSize=8,
                     textColor=INK, spaceAfter=2, leading=12, leftIndent=6),
    }


def _rule(color=BORDER): return HRFlowable(width="100%", thickness=0.5, color=color, spaceAfter=3)

def _section(title: str, S: dict):
    return [
        _rule(RED),
        Paragraph(title.upper(), ParagraphStyle(
            "sh", fontName="Helvetica-Bold", fontSize=9,
            textColor=RED, spaceAfter=2, letterSpacing=0.8
        )),
        Spacer(1, 1*mm),
    ]


def _pill(text: str, bg, fg=WHITE) -> str:
    """Inline colored pill — used in table cells."""
    return text  # plain text fallback (Paragraph handles colour via style)


def _meta_row(label: str, value: str, S: dict):
    return [Paragraph(label, S["dim"]), Paragraph(value, S["h3"])]


# ── Main generator ────────────────────────────────────────────────────────────
def generate_pdf(report_data: dict, output_path: Optional[str] = None) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=18*mm, rightMargin=18*mm,
        topMargin=14*mm, bottomMargin=14*mm,
        title="Kalvium Demo Audit Report",
        author="Kalvium Demo Audit Platform",
    )
    S     = _S()
    story = []

    # ── Extract top-level fields ──────────────────────────────────────────────
    score_data   = report_data.get("score", {})
    overall      = score_data.get("overall", 0)
    grade        = score_data.get("grade", "F")
    session_id   = report_data.get("session_id", "")[:8].upper()
    duration     = report_data.get("duration_seconds", 0)
    language     = report_data.get("language_detected", "en")
    speakers     = report_data.get("speakers_detected", [])
    created_at   = report_data.get("created_at", "")[:16].replace("T", "  ")
    rec_file     = Path(report_data.get("recording_file", "recording")).name
    adm_prob     = score_data.get("admission_probability",
                   report_data.get("intent", {}).get("admission_probability", 0))
    highlights   = report_data.get("coaching_highlights", [])
    strengths    = report_data.get("top_strengths", [])
    improvements = report_data.get("improvement_areas", [])
    talk         = report_data.get("talk_ratio", {})
    compliance   = report_data.get("script_compliance", {})
    objections   = report_data.get("objections", {})
    sentiment    = report_data.get("advanced_sentiment", report_data.get("sentiment", {}))
    cat_scores   = score_data.get("category_scores", {})
    llm_evals    = report_data.get("llm_evaluations", [])
    cat_coaching = report_data.get("category_coaching", {})
    utterances   = report_data.get("utterances", [])
    dur_m, dur_s = divmod(int(duration), 60)

    # ═══════════════════════════════════════════════════════════════════════════
    # PAGE 1 — COVER
    # ═══════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("KALVIUM DEMO AUDIT PLATFORM", S["cover_brand"]))
    story.append(Paragraph("Demo Call Evaluation Report", S["cover_title"]))
    story.append(_rule(RED))
    story.append(Spacer(1, 3*mm))

    # Score ring + metadata side by side
    prob_color = GREEN if adm_prob >= 0.6 else AMBER if adm_prob >= 0.4 else RED
    meta_inner = Table([
        _meta_row("SESSION ID",          session_id, S),
        _meta_row("FILE",                rec_file, S),
        _meta_row("DATE",                created_at, S),
        _meta_row("DURATION",            f"{dur_m}m {dur_s}s", S),
        _meta_row("LANGUAGE",            language.upper(), S),
        _meta_row("SPEAKERS",            ", ".join(speakers) or "—", S),
        [Paragraph("ADMISSION LIKELIHOOD", S["dim"]),
         Paragraph(f"{adm_prob*100:.0f}%", ParagraphStyle(
             "ap", fontName="Helvetica-Bold", fontSize=13, textColor=prob_color))],
    ], colWidths=[40*mm, 90*mm])
    meta_inner.setStyle(TableStyle([
        ("TOPPADDING",    (0,0),(-1,-1), 3),
        ("BOTTOMPADDING", (0,0),(-1,-1), 3),
        ("LINEBELOW",     (0,0),(-1,-2), 0.3, BORDER),
    ]))

    cover_table = Table(
        [[ScoreRing(overall, grade, size=95), meta_inner]],
        colWidths=[40*mm, 130*mm],
    )
    cover_table.setStyle(TableStyle([
        ("VALIGN",     (0,0),(-1,-1), "MIDDLE"),
        ("LEFTPADDING",(0,0),(0,-1),  0),
        ("LEFTPADDING",(1,0),(1,-1),  6*mm),
    ]))
    story.append(cover_table)
    story.append(Spacer(1, 4*mm))

    # Key finding highlight
    if highlights:
        hi_text = highlights[0]
        hi_bg = Table(
            [[Paragraph(f"★  {hi_text}", ParagraphStyle(
                "hib", fontName="Helvetica-Bold", fontSize=8.5,
                textColor=INK, leading=12))]],
            colWidths=[PAGE_W - 36*mm],
        )
        hi_bg.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,-1), RED_LIGHT),
            ("TOPPADDING",    (0,0),(-1,-1), 5),
            ("BOTTOMPADDING", (0,0),(-1,-1), 5),
            ("LEFTPADDING",   (0,0),(-1,-1), 8),
            ("ROUNDEDCORNERS",(0,0),(-1,-1), [4,4,4,4]),
        ]))
        story.append(hi_bg)
        story.append(Spacer(1, 3*mm))

    # Quick stats — single row table
    comp_rate = compliance.get("completion_rate", None)
    comp_str  = f"{comp_rate:.0f}%" if isinstance(comp_rate, (int, float)) else "—"
    stats = [
        ("Counsellor Talk",  f"{talk.get('counsellor_pct', 0):.0f}%"),
        ("Questions Asked",  str(talk.get('counsellor_questions', 0))),
        ("Script Completion",comp_str),
        ("Objections",       f"{objections.get('resolved',0)}/{objections.get('total_objections',0)} resolved"),
        ("Closing",          "Yes ✓" if compliance.get("closing_attempted") else "No  ✗"),
    ]
    stat_data  = [[Paragraph(k, S["dim"]) for k,_ in stats],
                  [Paragraph(v, S["h3"])  for _,v in stats]]
    stat_table = Table(stat_data, colWidths=[35*mm]*5)
    stat_table.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), GRAY_BG),
        ("TOPPADDING",    (0,0),(-1,-1), 5),
        ("BOTTOMPADDING", (0,0),(-1,-1), 5),
        ("LEFTPADDING",   (0,0),(-1,-1), 6),
        ("BOX",           (0,0),(-1,-1), 0.5, BORDER),
        ("LINEBEFORE",    (1,0),(-1,-1), 0.3, BORDER),
    ]))
    story.append(stat_table)
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════════
    # PAGE 2 — CATEGORY SCORES
    # ═══════════════════════════════════════════════════════════════════════════
    story += _section("Category Breakdown", S)

    if cat_scores:
        items = list(cat_scores.items())
    else:
        items = [(r.get("category_id", r.get("category","?")),
                  {"category": r.get("category","?"), "raw_score": r.get("score",5),
                   "weight": 0, "weighted_score": 0})
                 for r in llm_evals]

    # Split into 2 columns
    left_items  = items[:5]
    right_items = items[5:]

    def _bar_rows(item_list):
        rows = []
        for cat_id, cs in item_list:
            if isinstance(cs, dict):
                lbl = cs.get("category", cat_id)
                raw = cs.get("raw_score", 5.0)
                wt  = int(cs.get("weight", 0))
            else:
                lbl = cs.category; raw = cs.raw_score; wt = int(cs.weight)
            rows.append(HBar(lbl, raw, weight=wt, bar_w=75*mm))
            rows.append(Spacer(1, 2*mm))
        return rows

    two_col = Table(
        [[_bar_rows(left_items), _bar_rows(right_items)]],
        colWidths=[88*mm, 88*mm],
    )
    two_col.setStyle(TableStyle([
        ("VALIGN",      (0,0),(-1,-1), "TOP"),
        ("LEFTPADDING", (0,0),(-1,-1), 0),
        ("TOPPADDING",  (0,0),(-1,-1), 0),
    ]))
    story.append(two_col)
    story.append(Spacer(1, 4*mm))

    # Score legend
    legend = Table(
        [[Paragraph("● 7-10  Good", ParagraphStyle("lg",fontName="Helvetica",fontSize=7.5,textColor=GREEN)),
          Paragraph("● 5-6   Average", ParagraphStyle("la",fontName="Helvetica",fontSize=7.5,textColor=AMBER)),
          Paragraph("● 0-4   Needs Work", ParagraphStyle("lr",fontName="Helvetica",fontSize=7.5,textColor=RED))]],
        colWidths=[50*mm, 55*mm, 65*mm],
    )
    legend.setStyle(TableStyle([("TOPPADDING",(0,0),(-1,-1),0),("BOTTOMPADDING",(0,0),(-1,-1),0)]))
    story.append(legend)
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════════
    # PAGE 2b — PPT / DECK COVERAGE
    # ═══════════════════════════════════════════════════════════════════════════
    story += _section("PPT / Deck Coverage", S)
    deck_cov = report_data.get("deck_coverage")

    if not deck_cov or not deck_cov.get("deck_id"):
        story.append(Paragraph(
            "No deck was active for this session — deck coverage was not evaluated.",
            S["dim"]
        ))
        story.append(Spacer(1, 4*mm))
    else:
        def _pct_color(pct):
            if pct >= 80: return GREEN
            if pct >= 60: return AMBER
            return RED

        cov_pct    = deck_cov.get("coverage_score", 0) or 0
        must_cov   = deck_cov.get("must_covered", 0)
        must_tot   = deck_cov.get("must_total", 0)
        should_cov = deck_cov.get("should_covered", 0)
        should_tot = deck_cov.get("should_total", 0)

        deck_stats = [
            ("Deck", (deck_cov.get("deck_id") or "—")[:40]),
            ("Coverage Score", f"{cov_pct:.0f}%"),
            ("MUST Covered", f"{must_cov}/{must_tot}"),
            ("SHOULD Covered", f"{should_cov}/{should_tot}"),
        ]
        deck_stat_data  = [[Paragraph(k, S["dim"]) for k, _ in deck_stats],
                           [Paragraph(v, ParagraphStyle(
                               "dcv", fontName="Helvetica-Bold", fontSize=11,
                               textColor=_pct_color(cov_pct) if k == "Coverage Score" else INK))
                            for k, v in deck_stats]]
        deck_stat_table = Table(deck_stat_data, colWidths=[44*mm]*4)
        deck_stat_table.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,-1), GRAY_BG),
            ("TOPPADDING",    (0,0),(-1,-1), 5),
            ("BOTTOMPADDING", (0,0),(-1,-1), 5),
            ("LEFTPADDING",   (0,0),(-1,-1), 6),
            ("BOX",           (0,0),(-1,-1), 0.5, BORDER),
            ("LINEBEFORE",    (1,0),(-1,-1), 0.3, BORDER),
        ]))
        story.append(deck_stat_table)
        story.append(Spacer(1, 4*mm))

        # Per-criterion coverage table
        results = deck_cov.get("results", [])
        if results:
            imp_order = {"must": 0, "should": 1, "can": 2}
            results = sorted(results, key=lambda r: (imp_order.get(str(r.get("importance","")).lower(), 3),
                                                       not r.get("covered")))
            cov_data = [[
                Paragraph("TOPIC", ParagraphStyle("dh",fontName="Helvetica-Bold",fontSize=7.5,textColor=WHITE)),
                Paragraph("PRIORITY", ParagraphStyle("dh",fontName="Helvetica-Bold",fontSize=7.5,textColor=WHITE)),
                Paragraph("COVERED", ParagraphStyle("dh",fontName="Helvetica-Bold",fontSize=7.5,textColor=WHITE)),
            ]]
            for r in results:
                importance = str(r.get("importance", "")).upper()
                imp_color  = RED if importance == "MUST" else AMBER if importance == "SHOULD" else INK_DIM
                covered    = bool(r.get("covered"))
                cov_style  = ParagraphStyle("cvd", fontName="Helvetica-Bold", fontSize=7.5,
                                            textColor=GREEN if covered else RED)
                label = (r.get("label","") or "").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
                cov_data.append([
                    Paragraph(label[:80], S["dim"]),
                    Paragraph(importance, ParagraphStyle("ip", fontName="Helvetica-Bold", fontSize=7.5, textColor=imp_color)),
                    Paragraph("COVERED" if covered else "MISSED", cov_style),
                ])
            cov_table = Table(cov_data, colWidths=[110*mm, 28*mm, 28*mm])
            cov_table.setStyle(TableStyle([
                ("BACKGROUND",    (0,0),(-1,0), RED),
                ("ROWBACKGROUNDS",(0,1),(-1,-1),[WHITE, GRAY_BG]),
                ("TOPPADDING",    (0,0),(-1,-1), 4),
                ("BOTTOMPADDING", (0,0),(-1,-1), 4),
                ("LEFTPADDING",   (0,0),(-1,-1), 5),
                ("BOX",           (0,0),(-1,-1), 0.5, BORDER),
                ("LINEBELOW",     (0,0),(-1,-1), 0.3, BORDER),
                ("VALIGN",        (0,0),(-1,-1), "TOP"),
            ]))
            story.append(cov_table)
            story.append(Spacer(1, 4*mm))

        # Missed MUST / SHOULD callouts
        missed_must   = deck_cov.get("missed_must", [])
        missed_should = deck_cov.get("missed_should", [])
        if missed_must:
            story.append(Paragraph("MISSING — MUST-HAVE TOPICS", ParagraphStyle(
                "mmh", fontName="Helvetica-Bold", fontSize=8, textColor=RED, spaceAfter=2)))
            for m in missed_must[:8]:
                story.append(Paragraph(f"✗  {str(m).replace('&','&amp;')}", S["red"]))
            story.append(Spacer(1, 2*mm))
        if missed_should:
            story.append(Paragraph("MISSING — SHOULD-HAVE TOPICS", ParagraphStyle(
                "msh", fontName="Helvetica-Bold", fontSize=8, textColor=AMBER, spaceAfter=2)))
            for m in missed_should[:8]:
                story.append(Paragraph(f"✗  {str(m).replace('&','&amp;')}", S["amber"]))

    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════════
    # PAGE 3 — COACHING PLAYBOOK
    # ═══════════════════════════════════════════════════════════════════════════
    story += _section("Coaching Playbook", S)

    # Strengths + Improvements in 2 columns
    def _bullet_col(title, items, style, bullet):
        paras = [Paragraph(title, ParagraphStyle(
            "bch", fontName="Helvetica-Bold", fontSize=9,
            textColor=GREEN if bullet=="✓" else RED, spaceAfter=3
        ))]
        for item in items:
            clean = item.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
            paras.append(Paragraph(f"{bullet}  {clean}", style))
        return paras

    si_table = Table(
        [[_bullet_col("STRENGTHS",          strengths[:5],    S["green"], "✓"),
          _bullet_col("AREAS TO IMPROVE",   improvements[:6], S["red"],   "→")]],
        colWidths=[88*mm, 88*mm],
    )
    si_table.setStyle(TableStyle([
        ("VALIGN",      (0,0),(-1,-1), "TOP"),
        ("LEFTPADDING", (0,0),(-1,-1), 0),
        ("TOPPADDING",  (0,0),(-1,-1), 0),
    ]))
    story.append(si_table)
    story.append(Spacer(1, 4*mm))

    # Per-category coaching (compact)
    if cat_coaching:
        story += _section("Per-Category Coaching Tips", S)
        coaching_rows = []
        for cat_name, tips in cat_coaching.items():
            if not tips: continue
            tip = tips[0][:220] if tips else ""
            if not tip: continue
            coaching_rows.append([
                Paragraph(cat_name, S["h3"]),
                Paragraph(tip.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;"),
                          S["coaching"]),
            ])

        if coaching_rows:
            ctable = Table(coaching_rows, colWidths=[48*mm, 124*mm])
            ctable.setStyle(TableStyle([
                ("TOPPADDING",    (0,0),(-1,-1), 4),
                ("BOTTOMPADDING", (0,0),(-1,-1), 4),
                ("LEFTPADDING",   (0,0),(-1,-1), 4),
                ("VALIGN",        (0,0),(-1,-1), "TOP"),
                ("ROWBACKGROUNDS",(0,0),(-1,-1), [WHITE, GRAY_BG]),
                ("LINEBELOW",     (0,0),(-1,-1), 0.3, BORDER),
            ]))
            story.append(ctable)

    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════════
    # PAGE 4 — SENTIMENT & OBJECTIONS
    # ═══════════════════════════════════════════════════════════════════════════
    # Sentiment
    if sentiment:
        story += _section("Sentiment Analysis", S)
        sent_overall = sentiment.get("overall_sentiment", sentiment.get("overall", "—"))
        momentum     = sentiment.get("emotional_momentum", "—")
        eng_score    = sentiment.get("engagement_score", "—")
        narrative    = sentiment.get("gpt_sentiment_narrative", "")

        sent_stats = [
            ("Overall Sentiment",    sent_overall.upper()),
            ("Student Excited",      "Yes" if sentiment.get("student_excited") else "No"),
            ("Parent Convinced",     "Yes" if sentiment.get("parent_convinced") else "No"),
            ("Emotional Momentum",   str(momentum).upper()),
            ("Engagement Score",     str(eng_score)),
        ]
        sent_table = Table(
            [[Paragraph(k, S["dim"]), Paragraph(v, S["h3"])] for k, v in sent_stats],
            colWidths=[55*mm, 120*mm],
        )
        sent_table.setStyle(TableStyle([
            ("TOPPADDING",    (0,0),(-1,-1), 3),
            ("BOTTOMPADDING", (0,0),(-1,-1), 3),
            ("LEFTPADDING",   (0,0),(-1,-1), 4),
            ("ROWBACKGROUNDS",(0,0),(-1,-1), [WHITE, GRAY_BG]),
            ("LINEBELOW",     (0,0),(-1,-1), 0.3, BORDER),
        ]))
        story.append(sent_table)
        if narrative:
            story.append(Spacer(1, 2*mm))
            story.append(Paragraph(
                narrative[:400].replace("&","&amp;").replace("<","&lt;").replace(">","&gt;"),
                S["body"]
            ))
        story.append(Spacer(1, 4*mm))

    # Objections
    story += _section("Objection Analysis", S)
    obj_list = objections.get("objections", [])
    if obj_list:
        obj_data = [[
            Paragraph("OBJECTION", ParagraphStyle("oh",fontName="Helvetica-Bold",fontSize=7.5,textColor=WHITE)),
            Paragraph("CATEGORY",  ParagraphStyle("oh",fontName="Helvetica-Bold",fontSize=7.5,textColor=WHITE)),
            Paragraph("TIME",      ParagraphStyle("oh",fontName="Helvetica-Bold",fontSize=7.5,textColor=WHITE)),
            Paragraph("RESOLUTION",ParagraphStyle("oh",fontName="Helvetica-Bold",fontSize=7.5,textColor=WHITE)),
        ]]
        for o in obj_list:
            if not isinstance(o, dict): continue
            q   = o.get("resolution_quality", "?")
            qst = ParagraphStyle("qs", fontName="Helvetica-Bold", fontSize=7.5,
                                 textColor=GREEN if q=="well" else AMBER if q=="partial" else RED)
            obj_data.append([
                Paragraph((o.get("objection_text","")[:80]).replace("&","&amp;").replace("<","&lt;"), S["dim"]),
                Paragraph(o.get("category","?"), S["dim"]),
                Paragraph(str(o.get("timestamp","?")), S["dim"]),
                Paragraph(q.upper(), qst),
            ])
        ot = Table(obj_data, colWidths=[80*mm, 38*mm, 18*mm, 30*mm])
        ot.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,0), RED),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[WHITE, GRAY_BG]),
            ("TOPPADDING",    (0,0),(-1,-1), 4),
            ("BOTTOMPADDING", (0,0),(-1,-1), 4),
            ("LEFTPADDING",   (0,0),(-1,-1), 5),
            ("BOX",           (0,0),(-1,-1), 0.5, BORDER),
            ("LINEBELOW",     (0,0),(-1,-1), 0.3, BORDER),
            ("VALIGN",        (0,0),(-1,-1), "TOP"),
        ]))
        story.append(ot)
    else:
        story.append(Paragraph("No objections were detected in this call.", S["dim"]))

    # Turning points (compact)
    turning = sentiment.get("turning_points", []) if isinstance(sentiment, dict) else []
    if turning:
        story.append(Spacer(1, 4*mm))
        story += _section("Emotional Turning Points", S)
        for tp in turning[:5]:
            if not isinstance(tp, dict): continue
            desc = (tp.get("description","")).replace("&","&amp;").replace("<","&lt;")
            story.append(KeepTogether([
                Paragraph(
                    f"<b>{tp.get('timestamp_str','?')}</b>  "
                    f"{tp.get('from_sentiment','?')} → {tp.get('to_sentiment','?')}  "
                    f"[{tp.get('speaker','?')}]  <font color='#6b7280'>{tp.get('significance','').upper()}</font>",
                    S["h3"]),
                Paragraph(desc[:150], S["body"]),
                Spacer(1, 1.5*mm),
            ]))

    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════════
    # PAGE 5+ — TRANSCRIPT (English only, compact)
    # ═══════════════════════════════════════════════════════════════════════════
    story += _section("Full Transcript (English)", S)
    story.append(Paragraph(
        "Speaker labels: COUNSELLOR · STUDENT · PARENT · UNKNOWN",
        S["dim"]
    ))
    story.append(Spacer(1, 2*mm))

    # Strip opening audio/setup check utterances
    import re as _re
    _SETUP = [
        r"\bam i audible\b", r"\bcan you hear me\b", r"\bcan you see me\b",
        r"\baudio.*check\b", r"\bvideo.*check\b", r"\bturn on.*audio\b",
        r"\bturn on.*video\b", r"\bcheck.*check\b", r"\bnetwork.*issue\b",
        r"\bsignal.*outside\b", r"\binternet.*issue\b", r"\bscreen.*shar\b",
        r"^(hello[,\s!.]*)+$", r"^(hi[,\s!.]*)+$",
    ]
    def _is_setup_utt(utt_dict):
        txt = (utt_dict.get("english_text") or utt_dict.get("native_text") or "").lower().strip()
        if not txt: return True
        start = utt_dict.get("start_time", 999)
        if start < 180 and len(txt.split()) <= 5: return True
        return any(_re.search(p, txt, _re.IGNORECASE) for p in _SETUP)

    first_real = 0
    utt_list = [u if isinstance(u, dict) else u.__dict__ for u in utterances]
    for i, u in enumerate(utt_list[:8]):
        d = u if isinstance(u, dict) else vars(u)
        if _is_setup_utt(d): first_real = i + 1
        else: break
    utterances = utterances[first_real:]

    SPEAKER_COLORS = {
        "Counsellor": BLUE,
        "Student":    GREEN,
        "Parent":     AMBER,
        "Unknown":    INK_DIM,
    }

    tx_data = []
    for utt in utterances:
        if isinstance(utt, dict):
            speaker = utt.get("speaker", "Unknown")
            start   = utt.get("start_time", 0)
            text    = utt.get("english_text", "") or utt.get("native_text", "")
        else:
            speaker = getattr(utt.speaker, "value", str(utt.speaker))
            start   = utt.start_time
            text    = utt.english_text or utt.native_text

        if not text or not str(text).strip():
            continue

        mins, secs = divmod(int(start), 60)
        sp_color   = SPEAKER_COLORS.get(speaker, INK_DIM)
        sp_style   = ParagraphStyle("sp", fontName="Helvetica-Bold", fontSize=7.5,
                                    textColor=sp_color)
        clean_text = str(text)[:300].replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

        tx_data.append([
            Paragraph(f"{mins:02d}:{secs:02d}", S["dim"]),
            Paragraph(speaker.upper(), sp_style),
            Paragraph(clean_text, S["body"]),
        ])

    if tx_data:
        tx_table = Table(tx_data, colWidths=[12*mm, 22*mm, 138*mm])
        tx_table.setStyle(TableStyle([
            ("TOPPADDING",    (0,0),(-1,-1), 3),
            ("BOTTOMPADDING", (0,0),(-1,-1), 3),
            ("LEFTPADDING",   (0,0),(-1,-1), 3),
            ("VALIGN",        (0,0),(-1,-1), "TOP"),
            ("ROWBACKGROUNDS",(0,0),(-1,-1), [WHITE, GRAY_BG]),
            ("LINEBELOW",     (0,0),(-1,-1), 0.3, BORDER),
        ]))
        story.append(tx_table)
    else:
        story.append(Paragraph("No transcript available.", S["dim"]))

    # ── Footer on every page ──────────────────────────────────────────────────
    def _footer(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(INK_DIM)
        canvas.setFont("Helvetica", 6.5)
        canvas.drawString(18*mm, 8*mm,
            f"Kalvium Demo Audit Platform  ·  Session {session_id}  ·  "
            f"{datetime.utcnow().strftime('%Y-%m-%d')}")
        canvas.drawRightString(PAGE_W - 18*mm, 8*mm,
            f"Page {doc.page}")
        # Top red accent bar
        canvas.setFillColor(RED)
        canvas.rect(0, PAGE_H - 3, PAGE_W, 3, fill=1, stroke=0)
        canvas.restoreState()

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    pdf_bytes = buf.getvalue()

    if output_path:
        Path(output_path).write_bytes(pdf_bytes)
        logger.info(f"PDF saved → {output_path}")

    return pdf_bytes
