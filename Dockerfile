# Kalvium Audit Engine — Render deployment image.
#
# Pins the Python runtime and guarantees ffmpeg/ffprobe are present —
# both are required at runtime (audio/processor.py, video_analysis/
# frame_extractor.py, utils/url_downloader.py, and openai-whisper's own
# CLI all shell out to ffmpeg) and are NOT reliably available on a
# generic native buildpack, so this is the verifiable way to guarantee
# them rather than hoping the platform includes them.
FROM python:3.13-slim

# System dependencies:
#   ffmpeg        — audio/video extraction, chunking, frame extraction
#   build-essential — in case any Python dependency needs to compile a
#                     wheel for this platform (most ship prebuilt wheels;
#                     this is a safety net, not a known hard requirement)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first so this layer is cached across
# rebuilds that only change application code.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# Render injects its own $PORT at runtime and routes to it automatically —
# this default is only used for a local `docker run` without -e PORT=....
ENV PORT=8002
EXPOSE 8002

# No --reload (dev-only). $UPLOAD_DIR/$STORAGE_BACKEND/$DB_BACKEND/etc.
# all come from Render's environment variables, not from this image.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
