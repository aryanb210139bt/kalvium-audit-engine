"""
tests/test_diarizer_roles.py
Unit tests for diarization/diarizer.py::assign_roles_by_talktime — the
existing talk-time-ranking heuristic (previously inline only in
_diarize_pyannote), now extracted and reused by transcription/sarvam_batch's
diarized_transcript speaker_id mapping too. Pure function, no I/O.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.models import Speaker
from diarization.diarizer import assign_roles_by_talktime


def test_two_speakers_ranked_by_talktime():
    durations = {"0": 500.0, "1": 120.0}
    roles = assign_roles_by_talktime(durations)
    assert roles["0"] == Speaker.COUNSELLOR
    assert roles["1"] == Speaker.STUDENT


def test_three_speakers_ranked_counsellor_student_parent():
    durations = {"SPEAKER_02": 40.0, "SPEAKER_00": 900.0, "SPEAKER_01": 300.0}
    roles = assign_roles_by_talktime(durations)
    assert roles["SPEAKER_00"] == Speaker.COUNSELLOR
    assert roles["SPEAKER_01"] == Speaker.STUDENT
    assert roles["SPEAKER_02"] == Speaker.PARENT


def test_more_than_three_speakers_extras_are_unknown():
    durations = {"0": 400.0, "1": 300.0, "2": 200.0, "3": 100.0, "4": 50.0}
    roles = assign_roles_by_talktime(durations)
    assert roles["0"] == Speaker.COUNSELLOR
    assert roles["1"] == Speaker.STUDENT
    assert roles["2"] == Speaker.PARENT
    assert roles["3"] == Speaker.UNKNOWN
    assert roles["4"] == Speaker.UNKNOWN


def test_single_speaker():
    roles = assign_roles_by_talktime({"0": 60.0})
    assert roles["0"] == Speaker.COUNSELLOR


def test_empty_input():
    assert assign_roles_by_talktime({}) == {}
