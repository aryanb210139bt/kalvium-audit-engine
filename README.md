# Kalvium Audit Engine

AI-powered platform for auditing sales/counselling demo calls. Upload a recording
(file, Google Drive/Loom link, or CSV batch of links), and it transcribes,
scores, and coaches against a 9-category evaluation framework — with optional
add-ons for visual/participant analysis, deck-coverage alignment, and
per-associate performance tracking.

Live: **https://kalvium-audit-engine.onrender.com**

---

## Tech Stack

### Backend
- **Python 3.13**, **FastAPI** + **Uvicorn** (ASGI)
- **SQLite** (local dev) or **PostgreSQL** (production/Supabase) — selected per-module
  via `DB_BACKEND`; every store (`session_store`, `job_queue`, `sarvam_stt_jobs`,
  `video_audits`, `deck`, `activity_log`, `kalvium_auth`, `associate_roster`) has
  identical schemas on both backends (`db/postgres_schema.py` mirrors every SQLite
  `CREATE TABLE`)
- **Local disk** (dev) or **Cloudflare R2** (production) for persistent files
  (screenshots, Excel tracker, OAuth state) — independent of `DB_BACKEND`, set via
  `STORAGE_BACKEND`
- **Pydantic** / **pydantic-settings** for request/response models and `.env` config
- **WebSockets** (native FastAPI) for live pipeline progress streaming

### AI / ML services
| Purpose | Provider |
|---|---|
| Speech-to-text | **Sarvam AI Batch STT** (`saaras:v3`, async job — one job per recording, ≤2h/file, diarization + timestamps built in) is the default path (`pipeline_v3.py`'s `_batch_stt`). The older per-chunk cascade (**Sarvam** `saarika:v2.5` → **faster-whisper** → **openai-whisper**, `transcription/stt.py`) is kept intact and still used by transcript-paste/legacy flows. |
| 9-category audit + coaching | **OpenAI GPT-4o** |
| Video screenshot analysis (people count, slides) | **OpenAI GPT-4o** (vision, `response_format: json_object`) |
| Sentiment | **XLM-RoBERTa** (local, when `transformers` is installed) + GPT narrative — rule-based fallback otherwise |
| Speaker diarization | Sarvam's own `diarized_transcript` (speaker_id per segment) mapped to Counsellor/Student/Parent by total-talktime ranking, when available; stereo-channel / mono-energy acoustic heuristics as fallback; **pyannote.audio** optional (GPU + `HF_TOKEN`) |
| Deck coverage | Semantic embedding alignment between transcript and deck criteria (`deck/evaluation/semantic_aligner.py`), blended into the composite score alongside the behavioral (GPT-4o) score |

### Media processing (all local, no API cost)
- **FFmpeg / FFprobe** — audio extraction/conversion, frame extraction, metadata probing, >2h-recording splitting for the Batch STT size limit
- **Pillow** — screenshot resizing/compression, black-frame validation
- **pydub** — audio utilities

### Frontend
- Plain **HTML + vanilla JavaScript** — no build step, no framework
- **Tailwind CSS** (CDN) for styling, **Inter** / **JetBrains Mono** (Google Fonts)
- Native inline **SVG** for trend sparklines (no charting library)

### Integrations
- **Google Sheets API v4** (`google-api-python-client`) + **OAuth2** (`google-auth-oauthlib`) — push audit rows to a live tracker spreadsheet
- **openpyxl** — Excel tracker export/update-in-place, plus multi-sheet Audit Export (results + exact Tracker-format sheet + summary)
- **reportlab** — PDF report generation
- **gdown** — Google Drive downloads (handles the "too large to virus-scan" interstitial)

### Auth
- Custom session-cookie auth (`auth.py`) — **PBKDF2-HMAC-SHA256** password hashing (stdlib, no extra dependency), HttpOnly cookies, four roles: `admin`, `associate`, `uploader`, `viewer`

### Testing
- **pytest** / **pytest-asyncio** — **243 tests** across 16 files, covering the job queue,
  Sarvam Batch STT lifecycle + durable job store + smart-resume, speaker-role
  assignment, audio processing, video analysis pipeline, tracker auto-fill,
  history filtering/export, dashboard aggregation, and auth/roles. `tests/conftest.py`
  forces every test onto an isolated SQLite backend regardless of the real `.env`,
  so the suite can never touch production Postgres/R2.

---

## Architecture

```
Upload (file / Drive link / Loom / CSV batch)
        │
        ▼
  Job Queue (SQLite/Postgres) ── status: queued
        │  (explicit ▶ Start / Start All — never auto-starts)
        ▼
  Download / audio extraction (FFmpeg)
        │
        ├──► [optional] 5 screenshots (5/25/50/75/95%) + GPT-4o vision ──► tracker fields
        │
        ▼
  Sarvam Batch STT — one async job per recording (≤2h; split into
  valid-sized segments above that), diarization + timestamps included.
  Job id persisted immediately (sarvam_stt_jobs) — survives a restart;
  a retry checks the job's live status before ever resubmitting it.
        │
        ▼
  Speaker roles (talk-time ranking on Sarvam's diarization, or the
  acoustic stereo/mono-energy fallback) → Event detection → Sentiment
        │
        ▼
  Deck coverage alignment (semantic) → GPT-4o 9-category audit → Scoring → Coaching
        │
        ▼
  Report (PDF/Excel/JSON) → Google Sheets / Excel tracker push
        │
        ▼
  Audit History (filterable, card/list view, export) + Dashboard (aggregate KPIs)
```

### Key modules
| Path | Responsibility |
|---|---|
| `api/main.py` | All REST + WebSocket endpoints, background job orchestration |
| `pipeline_v3.py` | Audit pipeline orchestrator — Sarvam Batch STT (`_batch_stt`) by default, legacy chunked `_parallel_stt` kept intact but unused |
| `transcription/sarvam_batch.py` | Sarvam Batch STT lifecycle: initialise → upload → start → poll (capped backoff) → download → parse |
| `transcription/sarvam_job_store.py` | Durable per-session Sarvam job record — survives restarts; smart-resume support |
| `transcription/stt.py` | Legacy per-chunk STT cascade (Sarvam sync API → faster-whisper → openai-whisper) |
| `audio/processor.py` | Video/audio → WAV; chunking (legacy path) or single-file/`split_for_batch` (Batch STT path) |
| `diarization/diarizer.py` | Speaker labeling — acoustic fallback + shared talk-time role-ranking heuristic |
| `history_filters.py` | Single shared filter-to-SQL translation used identically by history, dashboard, and export |
| `reports/audit_export.py` | Excel/CSV export (filtered) — Audit Results + exact Tracker-format sheet + summary |
| `intelligence/participant_analyzer.py` | Per-role engagement scoring (SES/PES/CTR/SCS), attendance, flags |
| `audit/explainable_auditor.py` | 9-category GPT-4o audit with transcript evidence |
| `scoring/scorer_v2.py` | Weighted composite score + coaching |
| `deck/` | Semantic PPT/deck coverage alignment (own schema) |
| `video_analysis/` | Local frame extraction (FFmpeg/Pillow) + GPT-4o vision analysis |
| `job_queue.py` | Upload → queue → manual-start persistence, orphan recovery on restart |
| `session_store.py` | Completed/failed session + report persistence, filtered/paginated history queries |
| `activity_log.py` | Append-only audit-trail event log |
| `video_audit_store.py` | Video snapshot analysis job persistence + screenshots |
| `associate_roster.py` | Admin-managed list of valid associate names |
| `auth.py` | Users, sessions, roles |
| `reports/audit_excel_manager.py` | Tracker column schema, Excel read/write (update-in-place) |
| `reports/google_sheets_manager.py` | OAuth flow + Sheets read/write (update-in-place) |
| `reports/associate_analytics.py` | Per-associate and overall dashboard aggregation (filter-aware) |
| `utils/url_downloader.py` | Google Drive/Loom/direct URL → video or audio-only download |
| `db/pg.py`, `db/postgres_schema.py`, `db/backend.py` | Postgres connection pooling, mirrored schemas, backend selection |

### Frontend pages (`static/`)
| Page | Purpose |
|---|---|
| `login.html` | Real email/password sign-in |
| `dashboard.html` | Landing hub, role-aware |
| `audit.html` | New Audit — upload, job queue (slide-out drawer), live progress |
| `history.html` | Audit History — card/list view toggle, advanced filtering (date/associate/TL/lead stage/payment), Excel/CSV export, inline associate edit + delete (list view only); Audit Dashboard (aggregate KPIs) |
| `audit-detail.html` | Full single-audit view — score, categories, coaching, deck coverage, tracker editor, video snapshot evidence, activity trail, transcript |
| `associates.html` | Per-associate performance + trend |
| `manage-users.html` | Admin: create/manage accounts (4 roles) + associate roster, delete accounts |

---

## Setup

### 1. Python environment
```bash
source "/Users/apple/Desktop/demo audit/files/venv/bin/activate"
pip install -r requirements.txt
```

### 2. Environment variables (`.env`)
```env
OPENAI_API_KEY=...          # GPT-4o audit + coaching + video vision + translation
SARVAM_API_KEY=...          # Sarvam Batch STT (saaras:v3) + legacy saarika:v2.5
STT_PROVIDER=sarvam         # sarvam | faster-whisper | whisper | auto (legacy path only)
WHISPER_MODEL_SIZE=large-v3
DIARIZATION_PROVIDER=auto   # auto (real) | pyannote (needs GPU+HF_TOKEN) | mock (dev only)
HF_TOKEN=                   # only needed for pyannote

# Sarvam Batch STT tuning (sensible defaults — usually no need to set these)
SARVAM_BATCH_NUM_SPEAKERS=3
SARVAM_BATCH_POLL_INITIAL_SEC=10
SARVAM_BATCH_POLL_MAX_SEC=60
SARVAM_BATCH_POLL_BACKOFF=1.6
SARVAM_BATCH_POLL_TIMEOUT_SEC=10800

UPLOAD_DIR=...
MAX_UPLOAD_SIZE_MB=1000

DB_BACKEND=sqlite            # sqlite | postgres
DATABASE_URL=                # required only when DB_BACKEND=postgres
STORAGE_BACKEND=local        # local | r2
R2_ENDPOINT_URL=
R2_BUCKET_NAME=kalvium-audit-storage
R2_ACCESS_KEY_ID=
R2_SECRET_ACCESS_KEY=
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

## Deployment (Render)

`Dockerfile` (installs ffmpeg + Python deps) + `render.yaml` (Blueprint —
secrets stay in Render's own encrypted store, never in this repo). Current
phase: `DB_BACKEND=sqlite` + `STORAGE_BACKEND=r2`; flipping to
`DB_BACKEND=postgres` needs `scripts/migrate_sqlite_to_postgres.py` run first
(see `CLOUD_MIGRATION_PLAN.md`). Render's `CMD` launches
`uvicorn api.main:app --host 0.0.0.0 --port ${PORT}` directly — no `--reload`,
and logging is configured at module import time (not inside an
`if __name__ == "__main__":` guard) specifically so this launch path emits
`logger.info()` output correctly.

A deploy mid-audit will interrupt whatever's running on the old container;
`job_queue.recover_orphaned()` / `sarvam_job_store.recover_orphaned()` reset
anything left `processing`/`starting` back to a safely-restartable state on
the new container's startup — Resume/Retry/Start again from the queue.

---

## Testing
```bash
PYTHONPATH="$(pwd)" python3 -m pytest tests/ -v
```
243 tests. No live external API calls in the suite — Sarvam/OpenAI/Postgres/R2
are all mocked or forced onto an isolated local backend
(`tests/conftest.py`'s autouse fixture); external services are exercised
manually against real/deliberately-invalid inputs when needed.

---

## Feature highlights

- **Sarvam Batch STT** — a 90–120 minute recording is one async Sarvam job
  (or a handful for >2h recordings), not hundreds of individual chunk
  requests. Diarization and timestamps come back from Sarvam directly;
  speaker roles are assigned by talk-time ranking. The job id is persisted
  the instant it exists, so a retry/resume never blindly re-submits (and
  re-pays for) a job that already completed or is still running.
- **Upload → Queue → manual Start** — uploading never itself consumes
  Sarvam/OpenAI resources; only an explicit ▶ Start does. Soft-cancel for
  in-flight jobs also flips a cooperative flag the Sarvam poll loop checks.
- **Advanced Audit History filtering + export** — date range, associate, TL,
  lead stage, payment status — the exact same filter logic drives the list
  view, the dashboard, and the Excel/CSV export (one shared translation
  layer, never three reimplementations). Export includes a sheet matching
  the real Audit Tracker schema exactly.
- **List view with inline edit/delete** — a table view of Audit History
  alongside the card carousel, with associate-name editing and delete
  scoped only to that view.
- **Participant Intelligence** — per-role engagement scores (SES/PES/CTR/SCS),
  attendance, talk-time, and flags (parent absent, student passive,
  unresolved objections), transcript-only, non-breaking if it fails.
- **Video Snapshot & Participant Detection** (optional, opt-in) — 5
  screenshots at 5/25/50/75/95% of the video, local FFmpeg extraction +
  Pillow validation, GPT-4o vision per frame, deterministic rollup into
  tracker columns.
- **Deck coverage alignment** — semantic embedding comparison between the
  transcript and uploaded deck criteria, blended into the final composite
  score alongside the behavioral GPT-4o score.
- **CSV batch ingestion** — real CRM export format, full raw-row pass-through
  as lead-sheet evidence, auto-push to tracker.
- **Google Sheets / Excel tracker** — update-in-place by Session ID (never
  duplicates a row on re-push).
- **Role-based access** — admin / associate / uploader / viewer, enforced
  server-side (not just frontend redirects), admin-managed roster and user
  deletion.

## Known limitations
- No automated test coverage for `pipeline_v3.py`'s downstream audit/scoring
  logic itself (only the STT/diarization layer added recently is tested) —
  the core evaluation framework predates this test suite.
- Sarvam's `with_diarization` is a documented beta feature and can be
  omitted even when requested; the pipeline falls back to the existing
  acoustic diarizer in that case, which is less precise on non-stereo audio.
- A Render deploy landing mid-audit always interrupts whatever's running —
  inherent to swapping containers, not something application code can fully
  prevent. Recovery resets the job to a safely-restartable state, but the
  frontend doesn't yet surface "this job was interrupted, click Resume" —
  a dead WebSocket currently just freezes the last-seen progress silently.
- Single-machine, single-process deployment — concurrent real audits compete
  for the same CPU/memory.
- `--reload` dev mode restarts on every `.py` save, dropping in-memory
  session/progress state; orphan-recovery handles the resulting "stuck
  processing" row automatically on the next startup, but not the frontend
  freeze described above.
