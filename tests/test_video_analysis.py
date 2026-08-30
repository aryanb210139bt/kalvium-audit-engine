"""
tests/test_video_analysis.py
Focused unit tests for the Video Snapshot & Participant Detection add-on.
No repo-wide test suite exists yet (confirmed during the original PRD audit),
so these are the first tests in the project — scoped only to this feature's
pure logic (timestamp math, summary rollup, DB CRUD). They do not call
FFmpeg or OpenAI — those are exercised manually against a real/fake URL
per the "How to test" section of the feature's deliverables report.
"""
import json
import sqlite3
import sys
import threading
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from video_analysis import frame_extractor, vision_analyzer
import video_audit_store
from reports.audit_excel_manager import auto_fill_from_report


# ── frame_extractor.compute_timestamps (5/25/50/75/95% sampling) ───────────

def test_compute_timestamps_normal_video():
    points = frame_extractor.compute_timestamps(1200.0)  # 20-minute call
    assert len(points) == 5
    assert [p["percentage"] for p in points] == [5, 25, 50, 75, 95]
    assert [p["label"] for p in points] == ["5%", "25%", "50%", "75%", "95%"]
    seconds = [p["seconds"] for p in points]
    assert seconds == sorted(seconds)  # strictly increasing
    assert all(0 <= s < 1200.0 for s in seconds)
    # matches the worked example from the spec: 90min -> ~4:30, 22:30, 45:00, 67:30, 85:30
    ninety_min = frame_extractor.compute_timestamps(90 * 60)
    assert [round(p["seconds"] / 60, 1) for p in ninety_min] == [4.5, 22.5, 45.0, 67.5, 85.5]


def test_compute_timestamps_never_exactly_0_or_100_percent():
    points = frame_extractor.compute_timestamps(1000.0)
    seconds = [p["seconds"] for p in points]
    assert seconds[0] > 0.0
    assert seconds[-1] < 1000.0


def test_compute_timestamps_zero_duration():
    assert frame_extractor.compute_timestamps(0.0) == []


def test_compute_timestamps_very_short_video_dedupes():
    # Sub-5s video: percentage points collapse together at 1-decimal rounding
    # — must not return duplicate/near-duplicate timestamps.
    points = frame_extractor.compute_timestamps(2.0)
    seconds = [p["seconds"] for p in points]
    assert len(seconds) == len(set(seconds))


def test_extract_frame_missing_source_returns_false(tmp_path):
    # No ffmpeg subprocess should raise past this function for a bad input —
    # a failed frame is a gap in coverage, not a crash.
    missing = tmp_path / "does_not_exist.mp4"
    out = tmp_path / "out.jpg"
    ok = frame_extractor.extract_frame(missing, 1.0, out)
    assert ok is False
    assert not out.exists()


def test_extract_frame_validated_gives_up_after_all_retries_on_missing_source(tmp_path):
    missing = tmp_path / "does_not_exist.mp4"
    out = tmp_path / "out.jpg"
    ok, t = frame_extractor.extract_frame_validated(missing, 100.0, out, duration=200.0)
    assert ok is False


def test_is_frame_valid_rejects_solid_black_image(tmp_path):
    from PIL import Image
    black = tmp_path / "black.jpg"
    Image.new("RGB", (200, 200), color=(0, 0, 0)).save(black, "JPEG")
    assert frame_extractor._is_frame_valid(black) is False


def test_is_frame_valid_accepts_varied_image(tmp_path):
    from PIL import Image
    import random
    varied = tmp_path / "varied.jpg"
    img = Image.new("RGB", (200, 200))
    pixels = img.load()
    random.seed(0)
    for x in range(200):
        for y in range(200):
            pixels[x, y] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    img.save(varied, "JPEG")
    assert frame_extractor._is_frame_valid(varied) is True


def test_is_frame_valid_rejects_missing_file(tmp_path):
    assert frame_extractor._is_frame_valid(tmp_path / "nope.jpg") is False


def test_probe_metadata_missing_file_returns_partial_dict(tmp_path):
    meta = frame_extractor.probe_metadata(tmp_path / "nope.mp4")
    assert isinstance(meta, dict)
    assert "duration_seconds" not in meta  # ffprobe failed, nothing fabricated


def test_downscale_in_place_shrinks_oversized_image(tmp_path):
    from PIL import Image
    big = tmp_path / "big.jpg"
    Image.new("RGB", (1920, 1080), color="red").save(big, "JPEG")
    frame_extractor._downscale_in_place(big)
    with Image.open(big) as img:
        assert max(img.size) <= frame_extractor.MAX_SCREENSHOT_LONG_EDGE


def test_downscale_in_place_leaves_small_image_alone(tmp_path):
    from PIL import Image
    small = tmp_path / "small.jpg"
    Image.new("RGB", (400, 300), color="blue").save(small, "JPEG")
    before = small.stat().st_size
    frame_extractor._downscale_in_place(small)
    with Image.open(small) as img:
        assert img.size == (400, 300)  # untouched — already under the cap


# ── vision_analyzer.summarize_video ─────────────────────────────────────────

def test_summarize_video_empty():
    s = vision_analyzer.summarize_video([])
    assert s["frames_analyzed"] == 0
    assert "flags" in s


def test_summarize_video_all_failed():
    s = vision_analyzer.summarize_video([{"label": "X/5", "error": "boom"}] * 3)
    assert s["frames_analyzed"] == 0
    assert s["frames_failed"] == 3


def test_summarize_video_flags_no_people_no_slides():
    frames = [{"label": "X/5", "people_count": 0, "slides_presented": False}]
    s = vision_analyzer.summarize_video(frames)
    assert any("No people detected" in f for f in s["flags"])
    assert any("No slides/deck detected" in f for f in s["flags"])


def test_summarize_video_partial_failure_flag():
    frames = [
        {"label": "X/5", "people_count": 3, "slides_presented": True},
        {"label": "2X/5", "error": "vision call failed"},
    ]
    s = vision_analyzer.summarize_video(frames)
    assert s["frames_analyzed"] == 1
    assert s["frames_failed"] == 1
    assert s["slides_presented_in"] == "1/1"
    assert any("1 of 2 screenshots" in f for f in s["flags"])


def test_summarize_video_includes_tracker_fields():
    frames = [{"label": "X/5", "people_count": 3, "slides_presented": True}]
    s = vision_analyzer.summarize_video(frames)
    assert s["tracker_fields"]["Demo Attendees"] == "Student, Parent, Counsellor"
    assert s["tracker_fields"]["Deck Presentation"] == "Yes"


# ── derive_tracker_fields — the business rules from the spec ───────────────

def test_derive_tracker_fields_three_or_more_people():
    frames = [{"people_count": 3, "slides_presented": False}]
    tf = vision_analyzer.derive_tracker_fields(frames)
    assert tf["Demo Attendees"] == "Student, Parent, Counsellor"
    assert tf["Camera Status"] == "Both cameras on"


def test_derive_tracker_fields_two_people():
    frames = [{"people_count": 2, "slides_presented": False}]
    tf = vision_analyzer.derive_tracker_fields(frames)
    assert tf["Demo Attendees"] == "Student, Counsellor"
    assert tf["Camera Status"] == "Both cameras on"


def test_derive_tracker_fields_one_person():
    frames = [{"people_count": 1, "slides_presented": False}]
    tf = vision_analyzer.derive_tracker_fields(frames)
    assert tf["Demo Attendees"] == "Counsellor"
    assert tf["Camera Status"] == "Single camera on"


def test_derive_tracker_fields_zero_people():
    frames = [{"people_count": 0, "slides_presented": False}]
    tf = vision_analyzer.derive_tracker_fields(frames)
    assert tf["Demo Attendees"] == ""
    assert tf["Camera Status"] == ""


def test_derive_tracker_fields_verify_once_is_enough():
    # Per spec: "if these will verify only once then its a yes for all" — one
    # positive detection across the 5 samples settles the field, even if the
    # other 4 samples showed nothing (e.g. presenter turned off screen-share
    # for most of the call but showed slides briefly at one timestamp).
    frames = [
        {"people_count": 3, "slides_presented": True},
        {"people_count": 0, "slides_presented": False},
        {"people_count": 0, "slides_presented": False},
        {"people_count": 0, "slides_presented": False},
        {"people_count": 0, "slides_presented": False},
    ]
    tf = vision_analyzer.derive_tracker_fields(frames)
    assert tf["Demo Attendees"] == "Student, Parent, Counsellor"  # from the max, not an average
    assert tf["Deck Presentation"] == "Yes"
    assert tf["Screen-share Mode (Full screen / PiP)"] == "Full screen"


def test_derive_tracker_fields_no_slides_anywhere():
    frames = [{"people_count": 2, "slides_presented": False}, {"people_count": 1, "slides_presented": False}]
    tf = vision_analyzer.derive_tracker_fields(frames)
    assert tf["Deck Presentation"] == "No"
    assert tf["Screen-share Mode (Full screen / PiP)"] == ""


def test_derive_tracker_fields_all_frames_failed():
    tf = vision_analyzer.derive_tracker_fields([{"label": "X/5", "error": "boom"}])
    assert tf["Demo Attendees"] == ""
    assert tf["max_people_count"] == 0


# ── video_audit_store CRUD ───────────────────────────────────────────────────

def test_store_lifecycle(tmp_path, monkeypatch):
    # Point the store at a throwaway DB so this test never touches the real
    # data/video_audits.db or its screenshot directory.
    monkeypatch.setattr(video_audit_store, "DB_PATH", tmp_path / "test_video_audits.db")
    monkeypatch.setattr(video_audit_store, "_local", threading.local())
    video_audit_store.init_db()

    vid = str(uuid.uuid4())
    video_audit_store.create(vid, linked_session_id="sess-1", source_url="https://x",
                              label="Tester", actor="pytest", created_at="2026-01-01T00:00:00")
    rec = video_audit_store.get(vid)
    assert rec["status"] == "queued"
    assert rec["linked_session_id"] == "sess-1"

    video_audit_store.update_status(vid, "downloading")
    assert video_audit_store.get(vid)["status"] == "downloading"

    video_audit_store.set_duration(vid, 120.5)
    assert video_audit_store.get(vid)["duration_seconds"] == 120.5

    shots = [{"label": "X/5", "seconds": 10.0, "filename": "shot_X-5.jpg", "analysis": {"people_count": 2}}]
    summary = {"frames_analyzed": 1, "flags": []}
    video_audit_store.complete(vid, shots, summary, "2026-01-01T00:01:00")
    rec = video_audit_store.get(vid)
    assert rec["status"] == "completed"
    assert rec["screenshots"] == shots
    assert rec["summary"] == summary

    by_session = video_audit_store.get_by_session("sess-1")
    assert by_session["id"] == vid


def test_store_fail_path(tmp_path, monkeypatch):
    monkeypatch.setattr(video_audit_store, "DB_PATH", tmp_path / "test_video_audits2.db")
    monkeypatch.setattr(video_audit_store, "_local", threading.local())
    video_audit_store.init_db()

    vid = str(uuid.uuid4())
    video_audit_store.create(vid, linked_session_id="sess-2", source_url="https://x",
                              label="Tester", actor="pytest", created_at="2026-01-01T00:00:00")
    video_audit_store.fail(vid, "download 404", "2026-01-01T00:00:05")
    rec = video_audit_store.get(vid)
    assert rec["status"] == "failed"
    assert "404" in rec["error"]


def test_get_by_session_none_when_never_run(tmp_path, monkeypatch):
    monkeypatch.setattr(video_audit_store, "DB_PATH", tmp_path / "test_video_audits3.db")
    monkeypatch.setattr(video_audit_store, "_local", threading.local())
    video_audit_store.init_db()
    assert video_audit_store.get_by_session("never-existed") is None


# ── auto_fill_from_report — video evidence feeding the tracker columns ──────

def _minimal_report():
    return {"score": {"overall": 50, "grade": "C", "category_scores": {}},
            "deck_coverage": {}, "improvement_areas": [], "top_strengths": [],
            "duration_seconds": 0, "talk_ratio": {}, "participant_intelligence": {}}


def test_auto_fill_populates_tracker_fields_from_completed_video_analysis():
    video_rec = {
        "status": "completed",
        "summary": {"tracker_fields": {
            "Demo Attendees": "Student, Parent, Counsellor",
            "Camera Status": "Both cameras on",
            "Deck Presentation": "Yes",
            "Screen-share Mode (Full screen / PiP)": "Full screen",
        }},
    }
    row = auto_fill_from_report(_minimal_report(), {"video_analysis": video_rec, "lead_sheet": {}})
    assert row["Demo Attendees"] == "Student, Parent, Counsellor"
    assert row["Camera Status"] == "Both cameras on"
    assert row["Screen-share Mode (Full screen / PiP)"] == "Full screen"
    assert row["Deck Presentation"] == "Yes"  # video evidence overrides the deck-score guess


def test_auto_fill_leaves_video_columns_blank_when_not_run():
    row = auto_fill_from_report(_minimal_report(), {"video_analysis": None, "lead_sheet": {}})
    assert row["Demo Attendees"] == ""
    assert row["Camera Status"] == ""
    assert row["Screen-share Mode (Full screen / PiP)"] == ""


def test_auto_fill_ignores_video_analysis_still_in_progress():
    # Status not "completed" yet — must not fill partial/stale data.
    video_rec = {"status": "analyzing", "summary": {}}
    row = auto_fill_from_report(_minimal_report(), {"video_analysis": video_rec, "lead_sheet": {}})
    assert row["Demo Attendees"] == ""


# ── _run_preprocessed_pipeline — shared-download failure isolation ─────────
# Regression test for a real bug: a failure during the shared download step
# (before _preprocess_and_analyze_video even starts) used to leave the
# video_audit row stuck at "queued" forever, with no error recorded, while
# the audio session correctly failed. Caught via live testing against a
# real (deliberately broken) Drive link.

def test_shared_download_failure_marks_video_audit_failed_not_stuck_queued(tmp_path, monkeypatch):
    import api.main as main
    import activity_log
    from progress_tracker import ProgressTracker

    monkeypatch.setattr(video_audit_store, "DB_PATH", tmp_path / "test_pipeline_fail.db")
    monkeypatch.setattr(video_audit_store, "_local", threading.local())
    video_audit_store.init_db()

    # This test exercises the real failure path, which also calls
    # activity_log.log_event() — isolate that too, or every run of this
    # test writes real rows into data/activity_log.db (harmless-looking
    # locally, but it's exactly what turned into real Supabase pollution
    # once DB_BACKEND=postgres was in play; conftest.py's autouse fixture
    # now prevents the Postgres case, but this keeps local runs clean too).
    monkeypatch.setattr(activity_log, "DB_PATH", tmp_path / "test_activity_log.db")
    monkeypatch.setattr(activity_log, "_local", threading.local())
    activity_log.init_db()

    session_id = "sess-download-fail"
    video_audit_id = "va-download-fail"
    main._sessions[session_id] = {"session_id": session_id, "status": "downloading"}
    video_audit_store.create(video_audit_id, linked_session_id=session_id, source_url="https://x",
                              label="Tester", actor="pytest", created_at="2026-01-01T00:00:00")

    def _boom(url, dest_dir):
        raise ValueError("Google Drive download failed: simulated")

    monkeypatch.setattr("utils.url_downloader.download_recording", _boom)

    tracker = ProgressTracker(session_id)
    main._run_preprocessed_pipeline(session_id, "https://x", tmp_path, tracker,
                                     video_audit_id, "Tester", "pytest")

    assert main._sessions[session_id]["status"] == "failed"
    rec = video_audit_store.get(video_audit_id)
    assert rec["status"] == "failed"       # not stuck at "queued"/"downloading"
    assert rec["error"]                    # error message recorded, not blank


def test_successful_run_does_not_clobber_completed_video_status(tmp_path, monkeypatch):
    # Regression test for a real, 100%-reproducible bug: _run_preprocessed_pipeline
    # used to unconditionally call video_audit_store.update_status(video_audit_id,
    # "extracting_audio") right after _preprocess_and_analyze_video returned —
    # silently overwriting the 'completed' status (with screenshots_json/
    # summary_json already saved) that function had JUST set. Nothing later in
    # the success path ever touched video_audit_store again, so the row was
    # permanently stuck at "extracting_audio" even though the real work (GPT-4o
    # vision analysis of every screenshot) had already fully succeeded. Caught
    # live: every row in production stuck at "extracting_audio" already had
    # screenshots_json AND summary_json populated. The UI polled forever
    # showing "extracting_audio…" and never rendered the finished analysis.
    import api.main as main
    import activity_log
    from progress_tracker import ProgressTracker

    monkeypatch.setattr(video_audit_store, "DB_PATH", tmp_path / "test_success.db")
    monkeypatch.setattr(video_audit_store, "_local", threading.local())
    video_audit_store.init_db()
    monkeypatch.setattr(activity_log, "DB_PATH", tmp_path / "test_activity_log2.db")
    monkeypatch.setattr(activity_log, "_local", threading.local())
    activity_log.init_db()

    session_id = "sess-success"
    video_audit_id = "va-success"
    main._sessions[session_id] = {"session_id": session_id, "status": "downloading"}
    video_audit_store.create(video_audit_id, linked_session_id=session_id, source_url="https://x",
                              label="Tester", actor="pytest", created_at="2026-01-01T00:00:00")

    fake_video = tmp_path / "recording.mp4"
    fake_video.write_bytes(b"not a real video, just needs to exist for .stat()")

    def _fake_download(url, dest_dir):
        return fake_video

    def _fake_preprocess(video_path, vid, linked_session_id, label, actor):
        # Simulates the real function's own successful completion — it
        # already marks the row 'completed' with real data before returning.
        video_audit_store.complete(vid, [{"label": "5%", "filename": "shot.jpg"}],
                                    {"frames_analyzed": 1, "flags": []}, "2026-01-01T00:00:10")

    monkeypatch.setattr("utils.url_downloader.download_recording", _fake_download)
    monkeypatch.setattr(main, "_preprocess_and_analyze_video", _fake_preprocess)
    monkeypatch.setattr("utils.url_downloader.extract_audio_from_local_file", lambda *a, **kw: None)
    monkeypatch.setattr(main, "_run_pipeline", lambda *a, **kw: None)  # the rest of the real audit — not under test here

    tracker = ProgressTracker(session_id)
    main._run_preprocessed_pipeline(session_id, "https://x", tmp_path, tracker,
                                     video_audit_id, "Tester", "pytest")

    rec = video_audit_store.get(video_audit_id)
    assert rec["status"] == "completed"                 # NOT clobbered back to "extracting_audio"
    assert rec["screenshots"]                            # the real analysis results are still there
    assert rec["summary"]["frames_analyzed"] == 1
