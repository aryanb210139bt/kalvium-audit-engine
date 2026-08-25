"""
reports/excel_generator.py
Enterprise Excel report with multiple worksheets.

Sheets:
  1. Summary          — scores, metadata, admission likelihood
  2. Category Scores  — 19 categories with weights, scores, breakdown
  3. Sentiment        — speaker arcs, timeline, turning points
  4. Objections       — each objection with resolution quality
  5. Transcript       — full transcript with speaker, time, native, english
  6. Coaching         — per-category coaching playbook
  7. Talk Ratio       — communication balance metrics
"""
from __future__ import annotations
import io
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import openpyxl
from openpyxl import Workbook
from openpyxl.styles import (
    Font, PatternFill, Alignment, Border, Side, GradientFill,
)
from openpyxl.utils import get_column_letter
from openpyxl.chart import BarChart, Reference
from openpyxl.chart.series import DataPoint

logger = logging.getLogger(__name__)

# ── Colour constants (hex strings for openpyxl) ───────────────────────────────
C_DARK_BG  = "0D0F1A"
C_CARD     = "1E2035"
C_INDIGO   = "6366F1"
C_INDIGO_L = "818CF8"
C_SUCCESS  = "22C55E"
C_WARNING  = "FACC15"
C_DANGER   = "F87171"
C_NEUTRAL  = "94A3B8"
C_WHITE    = "E2E8F0"
C_DIM      = "64748B"


def _fill(hex_color: str) -> PatternFill:
    return PatternFill("solid", fgColor=hex_color)


def _font(bold=False, size=10, color=C_WHITE, italic=False) -> Font:
    return Font(bold=bold, size=size, color=color, italic=italic,
                name="Calibri")


def _align(h="left", v="center", wrap=False) -> Alignment:
    return Alignment(horizontal=h, vertical=v, wrap_text=wrap)


def _border_thin() -> Border:
    s = Side(style="thin", color="252840")
    return Border(left=s, right=s, top=s, bottom=s)


def _score_fill(score: float) -> PatternFill:
    if score >= 7: return _fill(C_SUCCESS)
    if score >= 5: return _fill(C_WARNING)
    return _fill(C_DANGER)


def _score_font(score: float) -> Font:
    return _font(bold=True, size=10, color="0D0F1A")


def _header_row(ws, cols: list[str], row: int = 1, bg: str = C_INDIGO):
    for col_i, col_name in enumerate(cols, 1):
        cell = ws.cell(row=row, column=col_i, value=col_name)
        cell.font = _font(bold=True, size=10, color=C_WHITE)
        cell.fill = _fill(bg)
        cell.alignment = _align("center")
        cell.border = _border_thin()


def _set_col_widths(ws, widths: dict[int, int]):
    for col, width in widths.items():
        ws.column_dimensions[get_column_letter(col)].width = width


def _freeze(ws, cell: str):
    ws.freeze_panes = cell


# ─────────────────────────────────────────────────────────────────────────────
# Sheet builders
# ─────────────────────────────────────────────────────────────────────────────

def _sheet_summary(wb: Workbook, report: dict):
    ws = wb.active
    ws.title = "📊 Summary"
    ws.sheet_view.showGridLines = False

    score_data  = report.get("score", {})
    overall     = score_data.get("overall", 0)
    grade       = score_data.get("grade", "F")
    intent      = report.get("intent", {})
    compliance  = report.get("script_compliance", {})
    talk        = report.get("talk_ratio", {})
    objections  = report.get("objections", {})
    sentiment   = report.get("advanced_sentiment", report.get("sentiment", {}))

    # Title
    ws.merge_cells("A1:F1")
    ws["A1"] = "KALVIUM DEMO AUDIT REPORT"
    ws["A1"].font = _font(bold=True, size=16, color=C_INDIGO_L)
    ws["A1"].fill = _fill(C_DARK_BG)
    ws["A1"].alignment = _align("center")

    ws.merge_cells("A2:F2")
    ws["A2"] = f"Session: {report.get('session_id', '')[:8].upper()}  |  {report.get('created_at', '')[:19]}"
    ws["A2"].font = _font(size=9, color=C_DIM)
    ws["A2"].fill = _fill(C_DARK_BG)
    ws["A2"].alignment = _align("center")

    # Score card
    rows = [
        ("", "METRIC", "VALUE", "BENCHMARK", "STATUS", ""),
        ("", "Overall Score", overall, "≥70 = B", "✓" if overall >= 70 else "✗", ""),
        ("", "Grade", grade, "B+ target", grade if grade in ("A+", "A", "B") else "Below target", ""),
        ("", "Admission Probability", f"{score_data.get('admission_probability', intent.get('admission_probability', 0))*100:.0f}%", "≥60%", "", ""),
        ("", "Counsellor Talk %", f"{talk.get('counsellor_pct', 0)}%", "<65%", "✓" if talk.get("counsellor_pct", 100) < 65 else "✗", ""),
        ("", "Discovery Questions", talk.get("counsellor_questions", 0), "≥5", "✓" if talk.get("counsellor_questions", 0) >= 5 else "✗", ""),
        ("", "Script Completion", f"{compliance.get('completion_rate', 0):.0f}%" if isinstance(compliance.get('completion_rate'), (int, float)) else "—", "≥80%", "", ""),
        ("", "Objections Resolved", f"{objections.get('resolved', 0)}/{objections.get('total_objections', 0)}", "100%", "", ""),
        ("", "Closing Attempted", "Yes" if compliance.get("closing_attempted") else "No", "Required", "✓" if compliance.get("closing_attempted") else "✗", ""),
        ("", "Student Excited", "Yes" if sentiment.get("student_excited") else "No", "Yes", "", ""),
        ("", "Parent Convinced", "Yes" if sentiment.get("parent_convinced") else "No", "Yes", "", ""),
        ("", "Emotional Momentum", sentiment.get("emotional_momentum", "—").title(), "Building", "", ""),
        ("", "Duration", f"{int(report.get('duration_seconds', 0)//60)}m {int(report.get('duration_seconds', 0)%60)}s", "30-60 min", "", ""),
        ("", "Language", report.get("language_detected", "—").upper(), "", "", ""),
    ]

    for row_i, row_data in enumerate(rows, 3):
        for col_i, val in enumerate(row_data, 1):
            cell = ws.cell(row=row_i, column=col_i, value=val)
            cell.fill = _fill(C_CARD if row_i % 2 == 0 else "252840")
            cell.font = _font(size=10, color=C_WHITE)
            cell.alignment = _align(wrap=True)
            cell.border = _border_thin()
            if col_i == 3 and row_i == 4:  # Overall score
                cell.font = _font(bold=True, size=14, color=C_SUCCESS if overall >= 70 else C_WARNING if overall >= 50 else C_DANGER)
            if col_i == 5:
                cell.font = _font(bold=True, color=C_SUCCESS if val == "✓" else C_DANGER if val == "✗" else C_WHITE)

    _header_row(ws, ["", "METRIC", "VALUE", "BENCHMARK", "STATUS", ""], row=3)
    _set_col_widths(ws, {1: 2, 2: 28, 3: 18, 4: 16, 5: 16, 6: 2})
    ws.row_dimensions[1].height = 30
    for r in range(4, len(rows) + 4):
        ws.row_dimensions[r].height = 18

    # Highlights
    row_start = len(rows) + 5
    ws.cell(row=row_start, column=2, value="KEY FINDINGS").font = _font(bold=True, size=12, color=C_INDIGO_L)
    for i, h in enumerate(report.get("coaching_highlights", []), row_start + 1):
        c = ws.cell(row=i, column=2, value=f"★ {h}")
        c.font = _font(color=C_WARNING)
        c.fill = _fill(C_CARD)
        ws.merge_cells(f"B{i}:F{i}")


def _sheet_categories(wb: Workbook, report: dict):
    ws = wb.create_sheet("📈 Category Scores")
    ws.sheet_view.showGridLines = False

    headers = ["CATEGORY", "WEIGHT %", "SCORE /10", "WEIGHTED", "BREAKDOWN", "TOP SUGGESTION"]
    _header_row(ws, headers)
    _set_col_widths(ws, {1: 28, 2: 12, 3: 12, 4: 12, 5: 50, 6: 50})
    _freeze(ws, "A2")

    score_data = report.get("score", {})
    llm_map = {r.get("category", r.get("category", "?")): r
               for r in report.get("llm_evaluations", [])}

    category_scores = score_data.get("category_scores", {})
    row = 2
    for cat_id, cs in category_scores.items():
        if isinstance(cs, dict):
            label   = cs.get("category", cat_id)
            raw     = cs.get("raw_score", 5.0)
            weight  = cs.get("weight", 0)
            weighted = cs.get("weighted_score", 0)
            breakdown = cs.get("breakdown", "")
        else:
            label   = cs.category
            raw     = cs.raw_score
            weight  = cs.weight
            weighted = cs.weighted_score
            breakdown = cs.breakdown

        llm = llm_map.get(label, {})
        suggestion = (llm.get("suggestions") or [""])[0] if isinstance(llm, dict) else ""

        row_data = [label, f"{weight}%", raw, round(weighted, 2), breakdown[:200], suggestion[:200]]
        for col_i, val in enumerate(row_data, 1):
            cell = ws.cell(row=row, column=col_i, value=val)
            cell.fill = _fill(C_CARD if row % 2 == 0 else "252840")
            cell.font = _font(size=9, color=C_WHITE)
            cell.alignment = _align(wrap=True)
            cell.border = _border_thin()
            if col_i == 3:
                cell.fill = _score_fill(raw)
                cell.font = _score_font(raw)
                cell.alignment = _align("center")

        ws.row_dimensions[row].height = 36
        row += 1

    # Score summary at bottom
    ws.cell(row=row + 1, column=1, value="TOTAL / OVERALL").font = _font(bold=True, color=C_INDIGO_L)
    overall_cell = ws.cell(row=row + 1, column=3, value=score_data.get("overall", 0))
    overall_cell.font = _font(bold=True, size=13, color=C_INDIGO_L)

    # Bar chart
    try:
        if row > 2:
            chart = BarChart()
            chart.type = "bar"
            chart.title = "Category Scores"
            chart.y_axis.title = "Score /10"
            chart.x_axis.title = "Category"
            chart.height = 12
            chart.width = 20

            data_ref = Reference(ws, min_col=3, min_row=1, max_row=row - 1)
            cats_ref = Reference(ws, min_col=1, min_row=2, max_row=row - 1)
            chart.add_data(data_ref, titles_from_data=True)
            chart.set_categories(cats_ref)
            chart.series[0].graphicalProperties.solidFill = C_INDIGO
            ws.add_chart(chart, f"H2")
    except Exception as e:
        logger.warning(f"Chart generation failed: {e}")


def _sheet_sentiment(wb: Workbook, report: dict):
    ws = wb.create_sheet("🎭 Sentiment")
    ws.sheet_view.showGridLines = False

    sentiment = report.get("advanced_sentiment", report.get("sentiment", {}))
    if not sentiment:
        ws["A1"] = "No advanced sentiment data available."
        return

    # Speaker arcs
    ws.merge_cells("A1:E1")
    ws["A1"] = "SPEAKER SENTIMENT ARCS"
    ws["A1"].font = _font(bold=True, size=12, color=C_INDIGO_L)

    _header_row(ws, ["SPEAKER", "OVERALL", "TRAJECTORY", "AVG SCORE", "CONVICTION / HESITATION SIGNALS"], row=2)
    arc_row = 3
    for arc_key in ("counsellor_arc", "student_arc", "parent_arc"):
        arc = sentiment.get(arc_key)
        if not arc:
            continue
        if isinstance(arc, dict):
            sp = arc.get("speaker", arc_key)
            overall = arc.get("overall_sentiment", "neutral")
            traj = arc.get("sentiment_trajectory", "stable")
            avg = arc.get("avg_sentiment_score", 0)
            conv = ", ".join(arc.get("conviction_signals", [])[:3])
            hes  = ", ".join(arc.get("hesitation_signals", [])[:3])
        row_vals = [sp.title(), overall.upper(), traj.title(), round(avg, 3), f"Conv: {conv} | Hes: {hes}"]
        for col_i, val in enumerate(row_vals, 1):
            cell = ws.cell(row=arc_row, column=col_i, value=val)
            cell.fill = _fill(C_CARD)
            cell.font = _font(size=9, color=C_SUCCESS if overall == "positive" else C_DANGER if overall == "negative" else C_WHITE)
            cell.border = _border_thin()
        arc_row += 1

    # Turning points
    tp_start = arc_row + 2
    ws.cell(row=tp_start, column=1, value="EMOTIONAL TURNING POINTS").font = _font(bold=True, size=11, color=C_INDIGO_L)
    _header_row(ws, ["TIME", "SPEAKER", "FROM", "TO", "SIGNIFICANCE", "TRIGGER"], row=tp_start + 1)

    tps = sentiment.get("turning_points", [])
    for i, tp in enumerate(tps, tp_start + 2):
        if isinstance(tp, dict):
            vals = [tp.get("timestamp_str", "?"), tp.get("speaker", "?"),
                    tp.get("from_sentiment", "?"), tp.get("to_sentiment", "?"),
                    tp.get("significance", "?").upper(), tp.get("trigger_text", "")[:60]]
        else:
            vals = [tp.timestamp_str, tp.speaker, tp.from_sentiment, tp.to_sentiment,
                    tp.significance.upper(), tp.trigger_text[:60]]
        for col_i, val in enumerate(vals, 1):
            cell = ws.cell(row=i, column=col_i, value=val)
            cell.fill = _fill("252840" if i % 2 == 0 else C_CARD)
            cell.font = _font(size=9)
            cell.border = _border_thin()

    # Narrative
    narrative = sentiment.get("gpt_sentiment_narrative")
    if narrative:
        nr_row = max(tp_start + 2 + len(tps), arc_row) + 2
        ws.cell(row=nr_row, column=1, value="GPT EMOTIONAL NARRATIVE").font = _font(bold=True, color=C_INDIGO_L)
        ws.merge_cells(f"A{nr_row+1}:F{nr_row+1}")
        c = ws.cell(row=nr_row + 1, column=1, value=narrative)
        c.font = _font(size=9, italic=True, color=C_WHITE)
        c.fill = _fill(C_CARD)
        c.alignment = _align(wrap=True)
        ws.row_dimensions[nr_row + 1].height = 48

    _set_col_widths(ws, {1: 12, 2: 16, 3: 14, 4: 14, 5: 14, 6: 50})


def _sheet_objections(wb: Workbook, report: dict):
    ws = wb.create_sheet("🚧 Objections")
    ws.sheet_view.showGridLines = False

    objections = report.get("objections", {})
    headers = ["OBJECTION", "CATEGORY", "TIMESTAMP", "ADDRESSED", "QUALITY", "OBJECTION TEXT"]
    _header_row(ws, headers)
    _set_col_widths(ws, {1: 4, 2: 30, 3: 14, 4: 12, 5: 14, 6: 60})

    objs = objections.get("objections", [])
    for row, o in enumerate(objs, 2):
        if isinstance(o, dict):
            quality = o.get("resolution_quality", "?")
            vals = [row - 1, o.get("category", "?"), o.get("timestamp", "?"),
                    "Yes" if o.get("was_addressed") else "No",
                    quality.upper(), o.get("objection_text", "")[:150]]
        else:
            quality = o.resolution_quality
            vals = [row - 1, o.category, o.timestamp,
                    "Yes" if o.was_addressed else "No",
                    quality.upper(), o.objection_text[:150]]

        for col_i, val in enumerate(vals, 1):
            cell = ws.cell(row=row, column=col_i, value=val)
            cell.fill = _fill(C_CARD if row % 2 == 0 else "252840")
            cell.font = _font(size=9)
            cell.alignment = _align(wrap=True)
            cell.border = _border_thin()
            if col_i == 5:
                cell.font = _font(bold=True, size=9,
                    color=C_SUCCESS if quality == "well" else C_WARNING if quality == "partial" else C_DANGER)
        ws.row_dimensions[row].height = 30

    # Summary
    sum_row = len(objs) + 3
    ws.cell(row=sum_row, column=1, value="SUMMARY").font = _font(bold=True, color=C_INDIGO_L)
    for label, val in [
        ("Total", objections.get("total_objections", 0)),
        ("Resolved Well", objections.get("resolved", 0)),
        ("Partial", objections.get("partially_resolved", 0)),
        ("Missed", objections.get("missed", 0)),
    ]:
        sum_row += 1
        ws.cell(row=sum_row, column=1, value=label).font = _font(color=C_DIM)
        ws.cell(row=sum_row, column=2, value=val).font = _font(bold=True)


def _sheet_transcript(wb: Workbook, report: dict):
    ws = wb.create_sheet("📝 Transcript")
    ws.sheet_view.showGridLines = False

    headers = ["#", "TIME", "SPEAKER", "LANGUAGE", "NATIVE TEXT", "ENGLISH TRANSLATION", "CONFIDENCE"]
    _header_row(ws, headers)
    _set_col_widths(ws, {1: 5, 2: 10, 3: 14, 4: 10, 5: 45, 6: 45, 7: 12})
    _freeze(ws, "A2")

    SPEAKER_COLORS = {
        "Counsellor": C_INDIGO_L,
        "Student":    C_SUCCESS,
        "Parent":     C_WARNING,
        "Unknown":    C_NEUTRAL,
    }

    utterances = report.get("utterances", [])
    for row, utt in enumerate(utterances, 2):
        if isinstance(utt, dict):
            speaker  = utt.get("speaker", "Unknown")
            start    = utt.get("start_time", 0)
            native   = utt.get("native_text", "")
            english  = utt.get("english_text", "")
            lang     = utt.get("language_detected", "en")
            conf     = utt.get("confidence", 1.0)
        else:
            speaker  = utt.speaker.value if hasattr(utt.speaker, "value") else str(utt.speaker)
            start    = utt.start_time
            native   = utt.native_text
            english  = utt.english_text
            lang     = utt.language_detected
            conf     = utt.confidence

        mins, secs = divmod(int(start), 60)
        sp_color = SPEAKER_COLORS.get(speaker, C_NEUTRAL)
        row_vals = [row - 1, f"{mins:02d}:{secs:02d}", speaker, lang.upper(),
                    native[:300], english[:300], f"{conf:.0%}"]

        for col_i, val in enumerate(row_vals, 1):
            cell = ws.cell(row=row, column=col_i, value=val)
            cell.fill = _fill(C_CARD if row % 2 == 0 else "252840")
            cell.alignment = _align(wrap=True, v="top")
            cell.border = _border_thin()
            if col_i == 3:
                cell.font = _font(bold=True, size=9, color=sp_color)
            else:
                cell.font = _font(size=9)
        ws.row_dimensions[row].height = 40


def _sheet_coaching(wb: Workbook, report: dict):
    ws = wb.create_sheet("🎯 Coaching")
    ws.sheet_view.showGridLines = False

    ws.merge_cells("A1:D1")
    ws["A1"] = "COUNSELLOR COACHING PLAYBOOK"
    ws["A1"].font = _font(bold=True, size=14, color=C_INDIGO_L)
    ws["A1"].fill = _fill(C_DARK_BG)
    ws["A1"].alignment = _align("center")

    _header_row(ws, ["CATEGORY", "SCORE", "PRIORITY", "COACHING RECOMMENDATION"], row=2)
    _set_col_widths(ws, {1: 28, 2: 10, 3: 12, 4: 80})

    category_coaching = report.get("category_coaching", {})
    score_data = report.get("score", {})
    cat_scores = score_data.get("category_scores", {})

    # If no category_coaching, build from LLM
    if not category_coaching:
        for r in report.get("llm_evaluations", []):
            if isinstance(r, dict):
                category_coaching[r.get("category", "?")] = r.get("suggestions", [])
            else:
                category_coaching[r.category] = r.suggestions

    # Merge with scores for priority sorting
    rows_to_write = []
    for cat_name, coaching_items in category_coaching.items():
        # Find score
        score = 5.0
        for cs in cat_scores.values():
            if isinstance(cs, dict):
                if cs.get("category") == cat_name:
                    score = cs.get("raw_score", 5.0)
            elif cs.category == cat_name:
                score = cs.raw_score
        priority = "HIGH" if score < 5 else "MEDIUM" if score < 7 else "LOW"
        coaching_text = " | ".join(coaching_items[:2]) if coaching_items else ""
        rows_to_write.append((cat_name, score, priority, coaching_text))

    rows_to_write.sort(key=lambda x: x[1])  # sort by score ascending (worst first)

    for row_i, (cat_name, score, priority, coaching_text) in enumerate(rows_to_write, 3):
        vals = [cat_name, f"{score:.1f}/10", priority, coaching_text]
        for col_i, val in enumerate(vals, 1):
            cell = ws.cell(row=row_i, column=col_i, value=val)
            cell.fill = _fill(C_CARD if row_i % 2 == 0 else "252840")
            cell.alignment = _align(wrap=True, v="top")
            cell.border = _border_thin()
            if col_i == 2:
                cell.font = _font(bold=True, size=10, color=C_SUCCESS if score >= 7 else C_WARNING if score >= 5 else C_DANGER)
            elif col_i == 3:
                color = C_DANGER if priority == "HIGH" else C_WARNING if priority == "MEDIUM" else C_SUCCESS
                cell.font = _font(bold=True, size=9, color=color)
            else:
                cell.font = _font(size=9)
        ws.row_dimensions[row_i].height = 48


def _sheet_talk_ratio(wb: Workbook, report: dict):
    ws = wb.create_sheet("🗣️ Talk Ratio")
    ws.sheet_view.showGridLines = False

    talk = report.get("talk_ratio", {})
    compliance = report.get("script_compliance", {})

    ws.merge_cells("A1:C1")
    ws["A1"] = "COMMUNICATION ANALYSIS"
    ws["A1"].font = _font(bold=True, size=13, color=C_INDIGO_L)
    ws["A1"].fill = _fill(C_DARK_BG)

    metrics = [
        ("Counsellor Talk %", f"{talk.get('counsellor_pct', 0)}%", "<65% ideal"),
        ("Student Talk %", f"{talk.get('student_pct', 0)}%", ">20% ideal"),
        ("Parent Talk %", f"{talk.get('parent_pct', 0)}%", ">10% ideal"),
        ("Total Questions", str(talk.get("total_questions_asked", 0)), ""),
        ("Counsellor Questions", str(talk.get("counsellor_questions", 0)), "≥5 ideal"),
        ("Interruptions", str(talk.get("interruption_count", 0)), "<5 ideal"),
        ("Dead Air (seconds)", str(talk.get("dead_air_sec", 0)), "<30s ideal"),
        ("Avg Response Latency", f"{talk.get('avg_response_latency_sec', 0):.1f}s", "<2s ideal"),
        ("Balance", "Balanced" if talk.get("is_balanced") else "Imbalanced", "Balanced ideal"),
        ("", "", ""),
        ("Script Stages Completed", "", ""),
    ]

    for i, (label, val, benchmark) in enumerate(metrics, 2):
        ws.cell(row=i, column=1, value=label).font = _font(size=10, color=C_DIM)
        ws.cell(row=i, column=2, value=val).font = _font(bold=True, size=10)
        ws.cell(row=i, column=3, value=benchmark).font = _font(size=9, color=C_DIM, italic=True)
        for col in range(1, 4):
            ws.cell(row=i, column=col).fill = _fill(C_CARD if i % 2 == 0 else "252840")
            ws.cell(row=i, column=col).border = _border_thin()

    # Script compliance
    comp_row = len(metrics) + 3
    ws.cell(row=comp_row, column=1, value="SCRIPT STAGE").font = _font(bold=True, color=C_INDIGO_L)
    ws.cell(row=comp_row, column=2, value="COMPLETED").font = _font(bold=True, color=C_INDIGO_L)

    if isinstance(compliance, dict):
        for r_i, (stage, done) in enumerate(compliance.items(), comp_row + 1):
            if stage == "completion_rate":
                continue
            ws.cell(row=r_i, column=1, value=stage.replace("_", " ").title()).font = _font(size=9)
            cell = ws.cell(row=r_i, column=2, value="✓" if done else "✗")
            cell.font = _font(bold=True, color=C_SUCCESS if done else C_DANGER)
            for col in range(1, 3):
                ws.cell(row=r_i, column=col).fill = _fill(C_CARD if r_i % 2 == 0 else "252840")

    _set_col_widths(ws, {1: 30, 2: 20, 3: 20})


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def generate_excel(report_data: dict, output_path: Optional[str] = None) -> bytes:
    """
    Generate a complete Excel audit report.

    Args:
        report_data: dict from DemoAuditReport.model_dump() + advanced_sentiment
        output_path: optional file path to save

    Returns:
        Excel file bytes
    """
    wb = Workbook()

    # Apply dark theme to all sheets
    def _style_tab(ws, color: str = C_INDIGO):
        ws.sheet_properties.tabColor = color

    _sheet_summary(wb, report_data)
    _style_tab(wb.active, C_INDIGO)

    _sheet_categories(wb, report_data)
    _style_tab(wb["📈 Category Scores"], C_SUCCESS)

    _sheet_sentiment(wb, report_data)
    _style_tab(wb["🎭 Sentiment"], C_WARNING)

    _sheet_objections(wb, report_data)
    _style_tab(wb["🚧 Objections"], C_DANGER)

    _sheet_transcript(wb, report_data)
    _style_tab(wb["📝 Transcript"], C_NEUTRAL)

    _sheet_coaching(wb, report_data)
    _style_tab(wb["🎯 Coaching"], C_INDIGO_L)

    _sheet_talk_ratio(wb, report_data)
    _style_tab(wb["🗣️ Talk Ratio"], C_CARD)

    buf = io.BytesIO()
    wb.save(buf)
    excel_bytes = buf.getvalue()

    if output_path:
        Path(output_path).write_bytes(excel_bytes)
        logger.info(f"Excel saved → {output_path}")

    return excel_bytes
