"""
video_analysis/vision_analyzer.py
GPT-4o vision analysis of a single extracted call-screenshot, plus a
rule-based rollup across all screenshots for one video. Follows the same
`OpenAI(api_key=settings.openai_api_key)` call convention used throughout
audit/ and intelligence/ — no new provider abstraction introduced.

Schema is deliberately narrow: per frame, the model only reports a people
COUNT and whether slides/a deck are being screen-shared — it is not asked to
guess *who* each person is. Per-person role attribution (parent vs student
vs counsellor) was the actual source of inaccuracy in earlier testing (the
model called two cameras "on" when only one tile was visible), so that
judgment call is removed from the model entirely. Attendee composition,
camera status, and screen-share mode for the tracker are instead derived
deterministically from the count + slide signal via fixed business rules
(see derive_tracker_fields) — "verify once, apply to the whole audit" per
spec: one clear positive across the 5 samples settles the field.

A failure analyzing one frame never raises past this module — it returns a
result dict with "error" set so the caller can keep going with the other
frames (partial coverage, not a failed job).
"""
from __future__ import annotations
import base64
import json
import logging

from config.settings import get_settings

logger = logging.getLogger(__name__)

VISION_MODEL = "gpt-4o"  # vision-capable; kept independent of settings.openai_model
                          # (audit text-model) so a future audit-model change
                          # can't silently break vision analysis.

_SYSTEM_PROMPT = """You are analyzing a single screenshot taken from a recorded \
online sales/counselling demo call (video call UI — e.g. Zoom/Meet/Teams grid). \
Identify what is visible and return ONLY a JSON object, no prose, matching \
exactly this shape:

{
  "people_count": <integer, total distinct people visible across all camera tiles>,
  "slides_presented": <true|false — a presentation/deck/document is being screen-shared>,
  "notes": "<one short factual sentence on anything notable, or empty string>",
  "confidence": {
    "people_count": <0.0-1.0>,
    "slides_presented": <0.0-1.0>
  }
}

Count every distinct person visible, including small camera tiles alongside a \
shared screen. Do not guess anyone's identity or role — only count how many \
people are visible. Base every field only on what is visibly present in the \
image; never invent people who aren't in the frame."""


def _image_to_data_url(image_path) -> str:
    data = image_path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def analyze_frame(image_path, label: str = "") -> dict:
    """
    Analyze one screenshot. Returns {"label", "people_count", "slides_presented",
    "notes", "confidence"}. On any failure, returns {"label", "error"} instead
    of raising — a bad frame is a gap in coverage, not a fatal error.
    """
    settings = get_settings()
    if not settings.openai_api_key:
        return {"label": label, "error": "OPENAI_API_KEY not configured"}

    try:
        from openai import OpenAI
        client = OpenAI(api_key=settings.openai_api_key)
        data_url = _image_to_data_url(image_path)

        resp = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": "Analyze this call screenshot."},
                    {"type": "image_url", "image_url": {"url": data_url, "detail": "low"}},
                ]},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=200,  # schema is tiny now — no need for the old 500-token budget
        )
        raw = resp.choices[0].message.content
        parsed = json.loads(raw)
        parsed["label"] = label
        # Normalize types defensively — a model that returns "true"/1 instead
        # of a real bool/int shouldn't break downstream int()/bool() logic.
        parsed["people_count"] = int(parsed.get("people_count") or 0)
        parsed["slides_presented"] = bool(parsed.get("slides_presented"))
        return parsed
    except Exception as e:
        logger.warning(f"analyze_frame failed for {label}: {e}")
        return {"label": label, "error": str(e)[:300]}


def derive_tracker_fields(frame_results: list[dict]) -> dict:
    """
    The audit-tracker-ready fields — exact column names from the tracker
    schema (reports/audit_excel_manager.py COLUMNS) — derived from the raw
    per-frame counts via fixed business rules, not left as "N/5 screenshots"
    ambiguity. Per spec: verify once across the 5 samples, that's the answer
    for the whole audit.

      max people_count >= 3  -> Student, Parent, Counsellor all attended
      max people_count == 2  -> Student, Counsellor attended
      max people_count == 1  -> Counsellor only
      max people_count == 0  -> nothing detected (blank — no evidence)

      max people_count > 1   -> both cameras were on (more than one live tile)
      max people_count == 1  -> single camera on
      max people_count == 0  -> blank

      slides_presented in ANY frame -> Deck Presentation = Yes,
                                       Screen-share Mode = Full screen
      slides_presented in NO frame  -> Deck Presentation = No, mode blank
    """
    ok = [f for f in frame_results if "error" not in f]
    if not ok:
        return {
            "Demo Attendees": "", "Camera Status": "", "Deck Presentation": "",
            "Screen-share Mode (Full screen / PiP)": "",
            "max_people_count": 0, "any_slides_detected": False,
        }

    max_people = max(f.get("people_count", 0) for f in ok)
    any_slides = any(f.get("slides_presented") for f in ok)

    if max_people >= 3:
        attendees = "Student, Parent, Counsellor"
        camera_status = "Both cameras on"
    elif max_people == 2:
        attendees = "Student, Counsellor"
        camera_status = "Both cameras on"
    elif max_people == 1:
        attendees = "Counsellor"
        camera_status = "Single camera on"
    else:
        attendees = ""
        camera_status = ""

    return {
        "Demo Attendees": attendees,
        "Camera Status": camera_status,
        "Deck Presentation": "Yes" if any_slides else "No",
        "Screen-share Mode (Full screen / PiP)": "Full screen" if any_slides else "",
        "max_people_count": max_people,
        "any_slides_detected": any_slides,
    }


def summarize_video(frame_results: list[dict]) -> dict:
    """
    Rule-based rollup across all successfully-analyzed frames (no extra LLM
    call needed) — human-readable coverage stats for the History/detail view,
    separate from derive_tracker_fields' single settled tracker values.
    """
    ok = [f for f in frame_results if "error" not in f]
    failed = [f for f in frame_results if "error" in f]

    if not ok:
        return {
            "frames_analyzed": 0,
            "frames_failed": len(failed),
            "flags": ["Could not analyze any screenshot — see per-frame errors"],
        }

    people_counts = [f.get("people_count", 0) for f in ok]
    slides_seen = [f for f in ok if f.get("slides_presented")]

    flags = []
    if max(people_counts, default=0) == 0:
        flags.append("No people detected in any analyzed screenshot")
    if not slides_seen:
        flags.append("No slides/deck detected in any analyzed screenshot")
    if failed:
        flags.append(f"{len(failed)} of {len(frame_results)} screenshots could not be analyzed")

    tracker_fields = derive_tracker_fields(frame_results)

    return {
        "frames_analyzed": len(ok),
        "frames_failed": len(failed),
        "avg_people_count": round(sum(people_counts) / len(people_counts), 1) if people_counts else 0,
        "max_people_count": tracker_fields["max_people_count"],
        "slides_presented_in": f"{len(slides_seen)}/{len(ok)}",
        "tracker_fields": tracker_fields,
        "flags": flags,
    }
