# Kalvium Audit Engine — Cloud Architecture & Account Planning (Phase 1)

**Status: planning only. No code, database, or deployment changes made.**

This document is the result of inspecting the full repository (not just the
README) — `api/`, `audit/`, `audio/`, `config/`, `deck/`, `intelligence/`,
`reports/`, `scoring/`, `transcription/`, `video_analysis/`, `utils/`,
`static/`, `tests/`, all DB/auth/env/API-integration/file-storage/temp-file/
background-processing/WebSocket code, and `.gitignore`/git history for
secrets exposure.

---

## 1. What the code actually does (not what the README implies)

### Live pipeline (confirmed via `pipeline_v3.py`'s own imports)
```
audio.processor.AudioProcessor        → FFmpeg conversion + chunking
transcription.stt.SpeechToText        → Sarvam → faster-whisper → openai-whisper cascade
diarization.diarizer.Diarizer         → stereo/mono-energy (default) or pyannote (opt-in)
intelligence.event_detector / analyzers / sentiment_v2
audit.explainable_auditor.ExplainableAuditor   → the ONE audit module actually used
audit.ppt_coverage, audit.evaluation_framework
scoring.scorer_v2
```

### Dead code found (imports `ANTHROPIC_API_KEY` / `GEMINI_API_KEY`, but never imported by `pipeline_v3.py`)
`audit/sales_auditor.py`, `audit/llm_auditor.py`, `audit/llm_auditor_v2.py`,
`audit/kalvium_layers.py`, `audit/kalvium_compliance.py`, `audit/buyer_intent.py`,
`audit/booking_authenticity.py`, `audit/engagement_analyzer.py`,
`audit/attendance_predictor.py`, `intelligence/call_intelligence.py`,
`intelligence/semantic_chunker.py`.

**Consequence: `ANTHROPIC_API_KEY` and `GEMINI_API_KEY` are not required.**
The `.env` template lists them as "optional" — inspection shows they're
currently unreachable from the live request path entirely. Don't create
Anthropic/Google Gemini accounts unless you plan to revive that code.

### GPU check
`torch` is installed (2.11.0) but only as a dependency of the Whisper
fallback path — it runs on CPU (`WHISPER_DEVICE=cpu` default). `pyannote`
(the one component that benefits from GPU) is **not installed** — the
diarizer's own code checks for it and falls back to the energy-based
diarizer automatically. **No GPU is required today.**

### Google Drive integration — two separate things, easy to conflate
1. **Downloading a submitted recording** (`utils/url_downloader.py`) uses
   `gdown` — scrapes a publicly-shared Drive file directly. **No Google
   Cloud project, no OAuth, no API key needed** for this to work; it only
   requires the source file to be shared "Anyone with the link."
2. **Pushing audit rows to a Google Sheet tracker**
   (`reports/google_sheets_manager.py`) is a real OAuth2 integration
   (`google-auth-oauthlib`, Sheets + Drive-metadata scopes) — this DOES need
   a Google Cloud project, an OAuth consent screen, and a client
   credentials JSON, already set up locally this session
   (`data/google_credentials.json`).

### Auth (inspected `auth.py`)
Custom, not JWT/OAuth: PBKDF2-HMAC-SHA256 password hashing (stdlib, no
extra package), server-side sessions in SQLite, HttpOnly cookie. Two roles
(`admin`, `associate`). This is fine to keep as-is for the MVP; production
hardening notes below.

### Storage today
- SQLite files under `data/` (12MB currently: sessions, activity log, auth,
  job queue, video-audit records, decks, associate roster).
- `uploads/` (currently 135MB) — per-session temp directories; the code
  deletes these after each pipeline run (`_cleanup_upload_dir`), but this
  local folder still shows real historical footprint.
- Generated PDFs/Excel/screenshots are written to local disk, served
  directly by FastAPI (`FileResponse`) — no object storage today.

### Background processing today
A single in-process `ThreadPoolExecutor(max_workers=4)` inside the same
FastAPI process. No Celery/RQ/Dramatiq/Redis anywhere in
`requirements.txt` or the code. This is the main thing that won't scale
past a handful of concurrent audits on one dyno/instance.

### Security finding (per your explicit request — flagging, not printing secrets)
- `.gitignore` correctly excludes `.env`, `data/`, `credentials.json`,
  `token.json`, `*.db` — none of these are tracked in git (`git ls-files`
  confirms). Good.
- **One real finding**: `auth.py` has a hardcoded seed admin password
  (plaintext, for the first-run admin account) directly in the tracked
  source file — this is pushed to GitHub right now. You already know this
  credential (it's the one documented in this project's own CLAUDE.md), but
  since the repo is on GitHub, **treat it as compromised and rotate it**
  before going anywhere near production — move it to an environment
  variable read at first-run instead of a literal in the file.

---

## 2. Data vs. large files — where each belongs

### A. Belongs in a database (→ PostgreSQL in production)
users/managers, associates (roster), audits (status, scores, category
scores, strengths/weaknesses/highlights/recommendations as JSON or
normalized rows), audit history, activity/action log, tracker push
records + diffs, video-audit job records (status, screenshot *paths*, not
the images themselves), job queue state, report *metadata* (not the PDF
bytes), permissions/roles, associate↔audit relationships, config that's
per-tenant rather than per-deploy.

Today these live across 7 separate SQLite files
(`sessions.db`, `activity_log.db`, `auth.db`, `job_queue.db`,
`video_audits.db`, `associate_roster.db`, `deck/decks.db`). All 7 map
cleanly to PostgreSQL tables/schemas — no redesign needed, just a
connection-string swap and a migration script per file.

### B. Must NOT go in PostgreSQL — needs object storage
Original videos (**never stored at all today — by design, and that should
stay true in production**), extracted screenshots/frames, generated
PDF/Excel reports, temporary processing files (raw audio, WAV chunks),
any Whisper model weights, the SQLite files' local disk footprint itself
if you don't migrate them.

---

## 3. Accounts / Services

Format per service: **Why / Used by / Cost / API key? / OAuth? / Card? / Credentials needed / Cheaper alternative / Recommendation**

### REQUIRED NOW (to run the current app in production, as-is)

**1. Sarvam AI**
- Why: primary STT for Indian languages — this is the actual product differentiator, not a fallback.
- Used by: `transcription/stt.py`
- Cost: paid, per-audio-minute — already have an account
- API key: yes · OAuth: no · Card: yes
- Credentials: `SARVAM_API_KEY`
- Cheaper alternative: faster-whisper (already the automatic fallback, free/local, lower accuracy on Indic languages)
- Recommend: keep — already load-bearing.

**2. OpenAI**
- Why: GPT-4o 9-category audit, coaching text, and the video-snapshot vision analysis
- Used by: `audit/explainable_auditor.py`, `video_analysis/vision_analyzer.py`, translation fallback
- Cost: paid, pay-per-token — this is your main variable cost driver
- API key: yes · OAuth: no · Card: yes
- Credentials: `OPENAI_API_KEY`
- Cheaper alternative: none that matches GPT-4o's structured-JSON reliability for this use case
- Recommend: keep — load-bearing, budget for it explicitly as volume grows.

**3. Render (hosting) — evaluate continuing**
- Why: already deployed there as a Web Service
- Used by: hosts `api/main.py`
- Cost: paid tier needed for anything beyond a toy demo (free tier sleeps, has no persistent disk)
- Card: yes
- Cheaper alternative: Railway/Fly.io are comparable at this scale; not worth switching for MVP
- Recommend: **keep for now** — see Option A/B below for exactly when this stops being enough.

**4. GitHub** (already have it)
- Why: source control + Render's deploy trigger
- Cost: free at this scale
- Recommend: keep, no action needed.

### REQUIRED FOR PRODUCTION (once managers/real audits are live)

**5. Managed PostgreSQL** (Render Postgres, or Neon/Supabase)
- Why: SQLite is single-file/single-writer — fine for one laptop, not for concurrent managers hitting the same data from a hosted process (and Render's disk is ephemeral on redeploy, so SQLite files vanish on every deploy today)
- Used by: replaces all 7 SQLite files
- Cost: ~$7–20/mo at small scale (Render Postgres Starter or Neon free→paid tier)
- Card: yes
- Credentials: `DATABASE_URL`
- Cheaper alternative: Neon has a genuinely usable free tier for MVP-scale traffic; Supabase similar
- Recommend: **Neon or Supabase for MVP** (free tier, no card needed to start), **Render Postgres if you want everything on one bill** — either is fine, don't overthink this one.

**6. Object storage — Cloudflare R2**
- Why: screenshots, generated PDFs/Excel, and (only if you ever decide to persist raw audio) audio files all need to survive a redeploy, which local disk on Render does not
- Used by: `video_analysis/`, `reports/pdf_generator.py`, `reports/excel_generator.py`
- Cost: R2 has no egress fees (unlike S3) and a generous free tier — genuinely the cheapest option here
- API key: yes (S3-compatible) · OAuth: no · Card: yes (billing on file, likely $0 at this volume)
- Credentials: `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET_NAME`
- Cheaper alternative: S3/GCS both charge egress; R2 is already the cheap option
- Recommend: **R2**, not S3 — same S3-compatible API, no egress cost when reports/screenshots get downloaded.

**7. A background worker + queue (Redis)**
- Why: FFmpeg/Whisper/GPT-4o calls are exactly the kind of work that must not run inline in the same process serving HTTP requests — this is the single most important architectural change for real concurrent volume, and it's the direct cause of the "server becomes unresponsive under load" issue already observed in local testing this session
- Used by: would replace the in-process `ThreadPoolExecutor` in `api/main.py`
- Cost: Render Key Value (their managed Redis) starts small; Upstash Redis has a serverless free tier
- Recommend: **Upstash Redis (free tier) + RQ** (simplest Python queue library, far less operational overhead than Celery for this codebase's needs) once you add a real worker process. Not needed for a single-manager MVP; required the moment "multiple managers, concurrent audits" becomes real.

**8. A separate Background Worker service (Render Background Worker, or a second Render Web Service running the RQ worker)**
- Why: video/audio/LLM processing needs to run on a process that can be scaled independently of the request-handling API, and can be given more CPU/RAM without paying for that on every API replica
- Cost: another small Render service, ~$7+/mo at minimum
- Recommend: add this at the same time as #7 — they're the same architectural change.

**9. Sentry (or Render's built-in logs, at first)**
- Why: you need to know when a real manager's audit fails in production, not find out from them
- Cost: Sentry free tier covers small-scale error tracking
- API key: yes (DSN) · Card: not required for free tier
- Credentials: `SENTRY_DSN`
- Recommend: add once a second real user exists; Render's own log tail is enough until then.

**10. Custom domain + DNS (any registrar, e.g. Cloudflare/Namecheap) + SSL**
- Why: `yourcompany.com` instead of `*.onrender.com` for anything manager-facing
- Cost: ~$10–15/yr for the domain; SSL is free (Render/Cloudflare both auto-provision Let's Encrypt)
- Recommend: get the domain once you're ready to hand a URL to a real manager, not before.

### OPTIONAL / LATER

**11. Google Cloud project + OAuth consent screen** — you already have this set up (for the Sheets tracker push). Nothing new to create; just note that in production this needs to move from "testing" to a verified OAuth consent screen if more than ~100 Google accounts will ever authorize it, and the consent screen's scopes need re-review.

**12. Email provider (Resend, Postmark, or SES)**
- Why: password resets, manager invitations, "your audit is ready" notifications — none of this exists in the code today (no email-sending code anywhere in the repo)
- Recommend: **not needed now** — the app has no password-reset flow or invitation flow built yet. Revisit when you build those features, not before.

**13. GPU compute (Modal / RunPod / cloud GPU)**
- Why: only relevant if you (a) turn on pyannote diarization for better accuracy, or (b) STT volume gets high enough that CPU Whisper fallback becomes a bottleneck
- Recommend: **do not create this account now.** Nothing in the current code requires it. Revisit only if Sarvam's cost/quota becomes a problem and you lean harder on local Whisper at volume.

**14. Managed workflow systems (Temporal, AWS Step Functions, etc.)**
- Why: only justified at a scale where the simple Redis+RQ worker queue (#7/#8) genuinely isn't enough
- Recommend: **not needed now, likely not needed for years** at this product's expected volume. A plain job queue is the right size.

---

## 4. Cost-conscious architecture — two options

### Option A — Minimum-cost MVP
*Small number of managers, low audit volume, dev/testing.*

```
Render Web Service (FastAPI, unchanged code)
        │
        ▼
  SQLite on Render's persistent disk (Render offers a small paid disk add-on)
  or Neon/Supabase Postgres free tier (recommended even at this stage —
  avoids a migration later, free tier is genuinely free)
        │
  Local disk for screenshots/reports (fine at low volume — Render's disk
  add-on, or R2 if you want zero risk of losing it on redeploy)
```
- Everything stays in the one FastAPI process, same `ThreadPoolExecutor`
  approach as today.
- **Monthly cost: ~$7–25** (one Render Web Service + free-tier Postgres).
- **Accept the limitation**: a handful of concurrent audits is fine; more
  than that will reproduce the "server becomes unresponsive" issue already
  seen locally.

### Option B — Production / scalable
*Multiple managers, concurrent audits, real volume.*

```
Render Web Service (FastAPI — auth, uploads, queue API, WebSocket progress)
        │
        ├──► Postgres (Neon/Supabase/Render) — all persistent data
        │
        ├──► Redis (Upstash) — job queue
        │
        ├──► Background Worker (separate Render service, RQ worker) ──► runs
        │      FFmpeg/Whisper/diarization/GPT-4o — the actual heavy lifting
        │
        └──► Cloudflare R2 — screenshots, PDFs, Excel, temp audio
```
- **What actually changes from A → B**: (1) SQLite → Postgres [connection
  string swap + migration script, `pipeline_v3.py`/audit logic untouched],
  (2) in-process executor → Redis+RQ with a separate worker service
  [`api/main.py`'s `loop.run_in_executor(...)` calls become `queue.enqueue(...)`],
  (3) local disk → R2 for anything that needs to survive a redeploy or be
  downloaded by a manager.
- **Monthly cost: roughly $40–100** depending on worker size and Postgres
  tier — still modest, no enterprise overhead.
- Video is still never permanently stored — the "download → 5 frames → GPT-4o
  vision → delete video" flow already built stays exactly as-is, it just
  runs on the worker instead of inline in the API process.

---

## 5. Final Checklists

### ACCOUNTS I SHOULD CREATE NOW
1. Sarvam AI — already have it, just confirm production quota/billing
2. OpenAI — already have it, confirm production rate limits
3. Render — already have it, confirm the paid tier you're on
4. GitHub — already have it

### ACCOUNTS I SHOULD CREATE LATER
1. Neon or Supabase (Postgres) — before your first real multi-manager rollout
2. Cloudflare R2 — same time as Postgres
3. Upstash Redis — when you add the background worker
4. Sentry — once a second real user exists
5. Domain registrar + DNS — when you're ready to hand out a real URL
6. Email provider (Resend/Postmark/SES) — only once you build password-reset/invitation features

### ACCOUNTS I DO NOT NEED
1. Anthropic — referenced only by dead code, not in the live pipeline
2. Google Gemini — same, dead code only
3. Any GPU provider (Modal/RunPod/cloud GPU) — nothing in the code requires it today
4. A separate Google Drive API/service-account setup for downloads — `gdown` already handles public-link downloads with zero Google Cloud config; your existing Google Cloud project is only for the Sheets push, already set up
5. Celery / managed workflow systems (Temporal etc.) — plain Redis+RQ is the right size for this product

### CREDENTIALS WE WILL NEED (only the ones actually relevant after inspection)
- `OPENAI_API_KEY`
- `SARVAM_API_KEY`
- `DATABASE_URL` (once on Postgres)
- `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET_NAME` (once on object storage)
- `REDIS_URL` (once on a queue)
- `SENTRY_DSN` (once monitoring is added)
- `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` — already have these (Sheets tracker OAuth), just carry them into the production environment's secret store, don't recommit `credentials.json`
- `HF_TOKEN` — only if pyannote diarization is ever turned on (not required today)
- Session/auth secret — currently derived internally by `auth.py`; worth promoting to an explicit `SESSION_SECRET` env var during the production hardening pass, alongside removing the hardcoded seed password from source.

Not needed: `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, any GPU provider key, any email provider key (yet).
