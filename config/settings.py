"""
config/settings.py
Pydantic Settings — reads from .env file automatically.
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenAI — entity extraction + translation fallback
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    translation_model: str = "gpt-4o-mini"

    # Anthropic — sales audit + coaching (primary LLM)
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-5"

    # Google — summarization (cheapest at scale)
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    # Sarvam AI — Indic language STT (Tamil, Kannada, Telugu, etc.)
    sarvam_api_key: str = ""

    # STT: "sarvam" | "whisper" | "faster-whisper" | "auto"
    stt_provider: str = "auto"
    whisper_model_size: str = "large-v3"
    whisper_device: str = "cpu"   # "cpu" | "cuda"

    # Diarization: "auto" | "pyannote" | "stereo" | "mock"
    diarization_provider: str = "auto"
    hf_token: str = ""   # HuggingFace token for Pyannote

    # Storage — TEMPORARY processing scratch space only (raw uploads/
    # downloads, audio chunks). Explicitly cleaned up after every pipeline
    # run (see api/main.py's _cleanup_upload_dir/_sweep_orphaned_uploads) —
    # never treated as persistent; on Render this is local disk and that's
    # fine, since nothing here needs to survive a restart. Persistent
    # application files (screenshots, the Excel tracker, Google OAuth
    # state) go through storage/r2.py instead — see STORAGE_BACKEND.
    #
    # Default is a portable relative path so this works out of the box on
    # any machine/container; set UPLOAD_DIR explicitly to override (this
    # project's own local .env does, for historical reasons).
    upload_dir: Path = Path("tmp_uploads")
    max_upload_size_mb: int = 1000

    # Database backend — "sqlite" (default, current behavior) or "postgres".
    # Only set db_backend="postgres" after scripts/migrate_sqlite_to_postgres.py
    # has succeeded (its verification report shows every table matching) —
    # see CLOUD_MIGRATION_PLAN.md. NEVER log/print database_url anywhere;
    # it's a full connection string with credentials.
    database_url: str = ""
    db_backend: str = "sqlite"

    # Object storage backend — "local" (default, current behavior) or "r2"
    # (Cloudflare R2, S3-compatible). Completely independent of db_backend —
    # you can run sqlite+r2, postgres+local, any combination. Only used by
    # storage/r2.py when storage_backend="r2". NEVER log/print
    # r2_access_key_id or r2_secret_access_key anywhere.
    storage_backend: str = "local"
    r2_endpoint_url: str = ""
    r2_bucket_name: str = "kalvium-audit-storage"
    r2_access_key_id: str = ""
    r2_secret_access_key: str = ""
    r2_region: str = "auto"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
