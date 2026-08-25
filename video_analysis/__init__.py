"""
video_analysis/
Optional add-on: Video Snapshot & Participant Detection.

Purely additive — does not touch audio/, transcription/, diarization/,
audit/, or pipeline_v3.py. Runs as a sibling job alongside the existing
audio-only audit pipeline, only when explicitly requested per-audit.
"""
