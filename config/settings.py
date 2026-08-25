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

    # Storage
    upload_dir: Path = Path("/Users/apple/Desktop/demo audit/demo audit without sarvam")
    max_upload_size_mb: int = 1000

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
