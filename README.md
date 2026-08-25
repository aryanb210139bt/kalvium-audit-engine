# Kalvium Audit Engine

AI-powered platform for auditing sales/counselling demo calls. Upload a recording
(file, Google Drive/Loom link, or CSV batch of links), and it transcribes,
scores, and coaches against a 9-category evaluation framework — with an
optional visual add-on that samples the video itself.

---

## Tech Stack

### Backend
- **Python 3.13**, **FastAPI** + **Uvicorn** (ASGI, `--reload` for dev)
- **SQLite** for all persistence — one small dedicated DB per concern, WAL mode,
  thread-local connections (no external database server to run)
- **Pydantic** / **pydantic-settings** for request/response models and `.env` config
- **WebSockets** (native FastAPI) for live pipeline progress streaming

### AI / ML services
| Purpose | Provider |
|---|---|
| Speech-to-text | **Sarvam AI** `saarika:v2.5` (Indian languages, primary) → **faster-whisper** (local CTranslate2, no API cost) → **openai-whisper** — cascading fallback |
| 9-category audit + coaching | **OpenAI GPT-4o** |
| Video screenshot analysis (people count, slides) | **OpenAI GPT-4o** (vision, `response_format: json_object`) |
| Sentiment | **XLM-RoBERTa** (local) + GPT narrative fallback |
| Speaker diarization | Stereo-channel / mono-energy heuristics (default, no ML) or **pyannote.audio** (optional, needs GPU + `HF_TOKEN`) |

### Media processing (all local, no API cost)
- **FFmpeg / FFprobe** — audio extraction, chunking, frame extraction, metadata probing
- **Pillow** — screenshot resizing/compression, black-frame validation
- **pydub** — audio utilities

### Frontend
- Plain **HTML + vanilla JavaScript** — no build step, no framework
- **Tailwind CSS** (CDN) for styling, **Inter** / **JetBrains Mono** (Google Fonts)
- Native `<canvas>`-free inline **SVG** for the trend sparkline (no charting library)

### Integrations
- **Google Sheets API v4** (`google-api-python-client`) + **OAuth2** (`google-auth-oauthlib`) — push audit rows to a live tracker spreadsheet
- **openpyxl** — Excel tracker export/update-in-place
- **reportlab** — PDF report generation
- **gdown** — Google Drive file downloads (also handles the "file too large to virus-scan" interstitial)

### Auth
- Custom session-cookie auth (`auth.py`) — **PBKDF2-HMAC-SHA256** password hashing (stdlib `hashlib`/`secrets`, no extra dependency), HttpOnly cookies, two roles (`admin`, `associate`)

### Testing
- **pytest** / **pytest-asyncio** — 43 tests covering the job queue, video analysis pipeline, and tracker auto-fill logic

---

## Architecture

```
Upload (file / Drive link / Loom / CSV batch)
        │
        ▼
  Job Queue (SQLite) ── status: queued
        │  (explicit ▶ Start / Start All — never auto-starts)
        ▼
  Download / audio extraction (FFmpeg)
        │
        ├──► [optional] 5 screenshots (5/25/50/75/95%) + GPT-4o vision ──► tracker fields
        │
        ▼
  Chunking (≤28s) → Sarvam/Whisper STT → Diarization
        │
        ▼
  GPT-4o 9-category audit → Scoring → Coaching
        │
        ▼
  Report (PDF/Excel/JSON) → Google Sheets / Excel tracker push
        │
        ▼
  Audit History (carousel) + Dashboard (aggregate KPIs)
```

### Key modules
| Path | Responsibility |
|---|---|
| `api/main.py` | All REST + WebSocket endpoints, background job orchestration |
| `pipeline_v3.py` | 7-step audit pipeline orchestrator |
| `audio/processor.py` | Video/audio → WAV → ≤28s chunks |
| `transcription/stt.py` | STT cascade (Sarvam → faster-whisper → openai-whisper) |
| `diarization/diarizer.py` | Speaker labeling |
| `audit/explainable_auditor.py` | 9-category GPT-4o audit with evidence |
| `scoring/scorer_v2.py` | Weighted composite score + coaching |
| `deck/` | Semantic PPT/deck coverage alignment (own SQLite DB) |
| `video_analysis/` | Local frame extraction (FFmpeg/Pillow) + GPT-4o vision analysis |
| `job_queue.py` | Upload → queue → manual-start persistence, orphan recovery |
| `session_store.py` | Completed/failed session + report persistence |
| `activity_log.py` | Append-only audit-trail event log |
| `video_audit_store.py` | Video snapshot analysis job persistence + screenshots |
| `associate_roster.py` | Admin-managed list of valid associate names |
| `auth.py` | Users, sessions, roles |
| `reports/audit_excel_manager.py` | Tracker column schema, Excel read/write (update-in-place) |
| `reports/google_sheets_manager.py` | OAuth flow + Sheets read/write (update-in-place) |
| `reports/associate_analytics.py` | Per-associate and overall dashboard aggregation |
| `utils/url_downloader.py` | Google Drive/Loom/direct URL → video or audio-only download |

### Frontend pages (`static/`)
| Page | Purpose |
|---|---|
| `login.html` | Real email/password sign-in |
| `dashboard.html` | Landing hub, role-aware (admin sees more) |
| `audit.html` | New Audit — upload, job queue (slide-out drawer), live progress |
| `history.html` | Audit History (horizontal carousel) + Audit Dashboard (aggregate KPIs) |
| `audit-detail.html` | Full single-audit view — score, categories, coaching, deck coverage, tracker editor, video snapshot evidence, activity trail, transcript |
| `associates.html` | Per-associate performance + trend |
| `manage-users.html` | Admin: create/manage accounts + associate roster |

---

## Setup

### 1. Python environment
```bash
source "/Users/apple/Desktop/demo audit/files/venv/bin/activate"
pip install -r requirements.txt
```

### 2. Environment variables (`.env`)
```env
OPENAI_API_KEY=...          # GPT-4o audit + coaching + video vision analysis
SARVAM_API_KEY=...          # Sarvam saarika:v2.5 STT (Indian languages)
STT_PROVIDER=sarvam         # sarvam | faster-whisper | whisper | auto
WHISPER_MODEL_SIZE=large-v3
DIARIZATION_PROVIDER=auto   # auto (real) | pyannote (needs GPU+HF_TOKEN) | mock (dev only)
HF_TOKEN=                   # only needed for pyannote
UPLOAD_DIR=...
MAX_UPLOAD_SIZE_MB=1000
```

### 3. Google Sheets (optional)
Upload OAuth `credentials.json` via the app's Sheets pill, complete the consent
flow, pick a spreadsheet + tab. Tokens/config persist under `data/`.

### 4. Run
```bash
bash start.sh
```
or manually:
```bash
PYTHONPATH="$(pwd)" uvicorn api.main:app --host 0.0.0.0 --port 8002 --reload
```
Open **http://localhost:8002/static/audit.html**

Login: `aryan@kalvium.com` / `kalvium123` (seeded on first run)

---

## Testing
```bash
PYTHONPATH="$(pwd)" python3 -m pytest tests/ -v
```
43 tests covering: job queue lifecycle (enqueue/start/cancel/orphan-recovery),
video frame extraction + validation, GPT-4o vision schema aggregation, and
tracker auto-fill from video evidence. No live API calls in tests — external
services are exercised manually against real/deliberately-invalid URLs.

---

## Feature highlights

- **Upload → Queue → manual Start** — uploading never itself consumes
  Sarvam/OpenAI/GPU resources; only an explicit ▶ Start does. Capacity-aware
  "Start All" reuses the existing `ThreadPoolExecutor` with zero extra
  scheduling code. Soft-cancel for in-flight jobs (can't safely hard-stop
  mid-transcription without touching the protected audit pipeline).
- **Video Snapshot & Participant Detection** (optional, opt-in) — 5 screenshots
  at 5/25/50/75/95% of the video, local FFmpeg extraction + Pillow
  resize/black-frame validation, GPT-4o vision per frame, deterministic
  business-rule rollup into tracker columns (Demo Attendees, Camera Status,
  Deck Presentation, Screen-share Mode). Single shared download with the
  audio pipeline — no duplicate downloads.
- **CSV batch ingestion** — real CRM export format (LeadSquared-style), full
  raw-row pass-through as lead-sheet evidence, auto-push to tracker.
- **Google Sheets / Excel tracker** — update-in-place by Session ID (never
  duplicates a row on re-push), auto-fill from report + video evidence.
- **Audit History carousel + Dashboard** — compact horizontal cards instead
  of one long page; aggregate KPIs, category performance, strengths/weakest
  categories, and trend over time, all computed from real session data.
- **Role-based access** — admin vs. associate, admin-managed roster,
  per-associate performance pages.

## Known limitations
- No automated test coverage for `pipeline_v3.py` itself (transcription/audit
  core) — CLAUDE.md marks it off-limits for modification, and no tests
  pre-existed for it.
- Video snapshot "cancel" is soft (UI-only) — the background job can't be
  safely interrupted mid-transcription without touching protected pipeline code.
- Single-machine, single-process deployment — concurrent real audits compete
  for the same CPU/memory; verified this can make the dev server transiently
  unresponsive under heavy concurrent load.
- `--reload` dev mode restarts on every `.py` save, which drops any
  in-memory session state; `job_queue`'s orphan-recovery handles the "stuck
  processing" side-effect of that automatically on the next startup.
