# Kalvium Audit Engine

Standalone AI-powered demo call audit platform. Lives at `/Users/apple/Desktop/Kalvium Audit Engine`.

## Start the server
```bash
bash start.sh
```
Or manually:
```bash
source "/Users/apple/Desktop/demo audit/files/venv/bin/activate"
PYTHONPATH="$(pwd)" uvicorn api.main:app --host 0.0.0.0 --port 8002 --reload
```
Open: **http://localhost:8002/static/audit.html**

## Login
- ID: `aryan@kalvium.com` | Password: `kalvium123`

## What it does
FastAPI backend + single-page frontend that:
1. Accepts demo call recordings (MP4/MP3/WAV/M4A/WebM) or Google Drive / Loom links
2. Transcribes with Sarvam AI saarika:v2.5 (Indian languages) → faster-whisper fallback
3. Runs 10-category GPT-4o audit (compliance, engagement, objection handling, closing, etc.)
4. Returns composite score (0–100 + grade) with explainable per-category reasoning
5. Participant Intelligence: per-role engagement scores, attendance, talk-time, flags

## Architecture
```
api/main.py               FastAPI app — all REST + WebSocket endpoints
pipeline_v3.py            7-step pipeline orchestrator
audio/processor.py        MP3/MP4/WAV → 25s WAV chunks
transcription/stt.py      STT cascade (Sarvam → faster-whisper → openai-whisper)
diarization/diarizer.py   Stereo / mono-energy / pyannote / mock speaker labels
intelligence/
  analyzers.py            TalkRatio, ScriptCompliance, Intent, Objection analyzers
  sentiment_v2.py         XLM-RoBERTa + GPT sentiment
  event_detector.py       Conversation event detection
  participant_analyzer.py NEW — per-role engagement scoring (V1, transcript-only)
audit/
  explainable_auditor.py  10-category GPT-4o audit with transcript evidence
  evaluation_framework.py EVALUATION_CATEGORIES definition
deck/                     Semantic PPT deck alignment (optional)
scoring/scorer_v2.py      Weighted composite score + coaching
config/models.py          All Pydantic data models
config/settings.py        Settings (reads .env)
static/audit.html         Full SPA frontend
utils/url_downloader.py   Google Drive / Loom / direct URL → MP3 via gdown + FFmpeg
```

## Key API endpoints
| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/v1/audit/upload` | Upload audio file |
| POST | `/api/v1/audit/upload-url` | Single Google Drive / direct URL |
| POST | `/api/v1/audit/upload-links` | Batch 1–20 links |
| POST | `/api/v1/audit/upload-transcript` | Upload .txt transcript |
| GET  | `/api/v1/audit/{id}` | Full report |
| GET  | `/api/v1/audit/{id}/participation` | Participant Intelligence |
| GET  | `/api/v1/audit/{id}/attendance` | Attendance record |
| GET  | `/api/v1/audit/{id}/participants` | Per-role profiles |
| GET  | `/api/v1/audit/{id}/download/pdf` | PDF report |
| GET  | `/api/v1/audit/{id}/download/excel` | Excel report |
| WS   | `/ws/{session_id}` | Real-time pipeline progress |

## Environment (.env)
```
OPENAI_API_KEY=...          GPT-4o for 10-category audit + sentiment narrative
SARVAM_API_KEY=...          Sarvam saarika:v2.5 STT (Indian languages)
STT_PROVIDER=sarvam         sarvam | faster-whisper | auto
WHISPER_MODEL_SIZE=large-v3
DIARIZATION_PROVIDER=auto   auto (real, stereo/mono-energy) | pyannote (best, needs HF_TOKEN+GPU) | mock (dev only, no real analysis)
HF_TOKEN=                   HuggingFace token (needed for pyannote diarization)
ANTHROPIC_API_KEY=...       Claude for audit layers (optional)
```

## Python venv
Shared venv at: `/Users/apple/Desktop/demo audit/files/venv/`
```bash
source "/Users/apple/Desktop/demo audit/files/venv/bin/activate"
```

## Participant Intelligence (V1 — built)
- `intelligence/participant_analyzer.py` — transcript-only analysis
- Computes: SES (Student Engagement Score), PES (Parent Engagement Score), CTR (Counsellor Talk Ratio), SCS (Stakeholder Coverage Score)
- Detects: parent absent/silent, student passive, unresolved objections, counsellor-dominated
- Falls back to rule-based narrative if OpenAI quota unavailable
- Runs as Step 7b after scoring — non-breaking (pipeline continues if it fails)

## Critical constraints
- **Do NOT modify**: transcription, chunking, STT pipeline, audit framework, report generation, output system
- All new features should be additive (new files / new fields in report_dict)
- The pipeline returns a plain `dict` (not Pydantic model) — add new keys to `report_dict` in `pipeline_v3.py`

## Input ingestion
- Google Drive → `gdown` download → FFmpeg audio extract → MP3
- Loom → CDN URL resolution → FFmpeg direct URL stream → MP3
- Direct MP4/video URL → FFmpeg direct URL stream → MP3
- Batch (1–20 links) → sequential queue with per-session WebSocket

## Known issues / notes
- `settings.upload_dir` points to old path — uploads go to `./uploads/` relative to project root
- pyannote diarization needs `HF_TOKEN` + GPU; mono-energy is the production default
- OpenAI quota may be exhausted — participant narrative falls back to rule-based automatically
- `--reload` flag means server auto-restarts on any `.py` file change
