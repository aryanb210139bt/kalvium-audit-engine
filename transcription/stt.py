"""
transcription/stt.py
Steps 3 & 5: Speech-to-text + English translation.

STT provider cascade (auto mode, tried in order):
  1. faster-whisper (ctranslate2 int8) — best Hindi/Hinglish, already installed
  2. Sarvam saarika:v2.5             — good for pure South-Indian languages
  3. openai-whisper (local fallback)  — last resort

Force a specific provider via STT_PROVIDER env var:
  "faster-whisper" | "sarvam" | "whisper" | "auto" (default)

Model size is controlled by WHISPER_MODEL_SIZE (default: medium).
  tiny / base  — fast, poor Hindi quality
  medium       — recommended: good Hindi/Hinglish, manageable CPU speed
  large-v3     — best accuracy, slow on CPU (~10× real-time)
"""
from __future__ import annotations
import logging
import re
import uuid
from pathlib import Path
from typing import Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

from config.models import AudioChunk, Speaker, Utterance
from config.settings import get_settings

logger   = logging.getLogger(__name__)
settings = get_settings()

SARVAM_LANGUAGES: dict[str, str] = {
    "hi-IN": "Hindi",
    "ta-IN": "Tamil",
    "te-IN": "Telugu",
    "kn-IN": "Kannada",
    "ml-IN": "Malayalam",
    "mr-IN": "Marathi",
    "bn-IN": "Bengali",
    "gu-IN": "Gujarati",
    "pa-IN": "Punjabi",
    "en-IN": "English (Indian)",
}

SARVAM_STT_URL   = "https://api.sarvam.ai/speech-to-text"
SARVAM_TRANS_URL = "https://api.sarvam.ai/translate"

# Whisper language codes for forced decoding
LANG_CODE_TO_WHISPER: dict[str, str] = {
    "hi-IN": "hi", "ta-IN": "ta", "te-IN": "te", "kn-IN": "kn",
    "ml-IN": "ml", "mr-IN": "mr", "bn-IN": "bn", "gu-IN": "gu",
    "pa-IN": "pa", "en-IN": "en",
}


# ── Hallucination detection ────────────────────────────────────────────────────

def _is_hallucination(text: str) -> bool:
    """
    Detect common transcription hallucinations:
    - Empty / too short
    - Known YouTube/subtitle filler phrases
    - Single character repeated 6+ times  (e.g. ◆◆◆◆◆◆, आपके repeated as chars)
    - Short sequence repeated 4+ times    (e.g. "आप " × 8)
    - Only numbers/dots/spaces            (e.g. "3. 3. 2. 2. 2.")
    - Word-level bigram repetition >55%
    """
    t = text.strip()
    if not t or len(t) < 4:
        return True

    tl = t.lower()
    for phrase in ("thank you for watching", "subscribe to our channel",
                   "please like and subscribe", "www.", "subtitles by",
                   "transcribed by", "amara.org"):
        if phrase in tl:
            return True

    # Single Unicode char repeated 6+ times
    if re.search(r'(.)\1{5,}', t):
        return True

    # Short sequence (2-8 chars) repeated 4+ consecutive times
    if re.search(r'(.{2,8})\1{3,}', t):
        return True

    # Only numbers, dots, dashes, spaces (e.g. "3. 3. 2. 2.")
    if re.fullmatch(r'[\d\s.,\-]+', t):
        return True

    # Word-level bigram repetition
    words = tl.split()
    if len(words) > 8:
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
        if len(set(bigrams)) / len(bigrams) < 0.45:
            return True

    return False


# ── Main STT class ─────────────────────────────────────────────────────────────

class SpeechToText:
    """
    Transcribes audio chunks and translates to English.
    Cascade: faster-whisper → Sarvam → local openai-whisper.
    """

    def __init__(self):
        self._fw_model      = None   # faster-whisper WhisperModel
        self._ow_model      = None   # openai-whisper model

    def transcribe_chunks(
        self,
        chunks: list[AudioChunk],
        speaker_map: Optional[dict[int, Speaker]] = None,
    ) -> list[Utterance]:
        utterances: list[Utterance] = []
        for chunk in chunks:
            speaker = speaker_map.get(chunk.chunk_id, Speaker.UNKNOWN) if speaker_map else Speaker.UNKNOWN
            utt = self._transcribe_chunk(chunk, speaker)
            if utt and utt.native_text.strip():
                utterances.append(utt)
        logger.info(f"Transcribed {len(utterances)}/{len(chunks)} non-empty chunks")
        return utterances

    # ── Routing ───────────────────────────────────────────────────────────────

    def _transcribe_chunk(self, chunk: AudioChunk, speaker: Speaker) -> Optional[Utterance]:
        provider = settings.stt_provider.lower()

        try:
            if provider == "faster-whisper":
                return self._transcribe_faster_whisper(chunk, speaker)

            if provider == "sarvam":
                utt = self._transcribe_sarvam(chunk, speaker)
                return utt or self._transcribe_faster_whisper(chunk, speaker)

            if provider == "whisper":
                return self._transcribe_openai_whisper_local(chunk, speaker)

            # "auto": faster-whisper → Sarvam → openai-whisper
            utt = self._transcribe_faster_whisper(chunk, speaker)
            if utt:
                return utt
            logger.debug(f"Chunk {chunk.chunk_id}: faster-whisper failed, trying Sarvam")

            if settings.sarvam_api_key:
                utt = self._transcribe_sarvam(chunk, speaker)
                if utt:
                    return utt
                logger.debug(f"Chunk {chunk.chunk_id}: Sarvam failed, trying local whisper")

            return self._transcribe_openai_whisper_local(chunk, speaker)

        except Exception as exc:
            logger.warning(f"Transcription failed for chunk {chunk.chunk_id}: {exc}")
            return None

    # ── faster-whisper (CTranslate2 int8) ────────────────────────────────────

    def _load_faster_whisper(self):
        if self._fw_model is None:
            from faster_whisper import WhisperModel
            size    = getattr(settings, "whisper_model_size", "medium")
            device  = getattr(settings, "whisper_device", "cpu")
            compute = "int8" if device == "cpu" else "float16"
            logger.info(f"Loading faster-whisper {size} ({device}/{compute})…")
            self._fw_model = WhisperModel(size, device=device, compute_type=compute)
            logger.info("faster-whisper ready")
        return self._fw_model

    def _transcribe_faster_whisper(
        self, chunk: AudioChunk, speaker: Speaker,
        language: Optional[str] = None,
    ) -> Optional[Utterance]:
        try:
            model = self._load_faster_whisper()
            wav   = str(Path(chunk.file_path))

            segments, info = model.transcribe(
                wav,
                language=language,           # None = auto-detect
                beam_size=5,
                best_of=5,
                temperature=0.0,
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 300},
                word_timestamps=False,
                no_speech_threshold=0.5,
            )

            # Consume the generator — segments are lazy
            full_text  = ""
            total_prob = 0.0
            seg_count  = 0
            for seg in segments:
                txt = seg.text.strip()
                if not _is_hallucination(txt):
                    full_text  += (" " if full_text else "") + txt
                    total_prob += seg.avg_logprob
                    seg_count  += 1

            if not full_text.strip() or _is_hallucination(full_text):
                return None

            avg_logprob = total_prob / max(seg_count, 1)
            confidence  = round(min(1.0, max(0.0, 1.0 + avg_logprob / 5)), 3)

            detected_lang = getattr(info, "language", "en") or "en"
            lang_code     = f"{detected_lang}-IN" if "-" not in detected_lang else detected_lang

            english_text = (
                full_text if detected_lang.startswith("en")
                else self._translate_to_english(full_text, lang_code)
            )

            return Utterance(
                utterance_id=str(uuid.uuid4()),
                speaker=speaker,
                start_time=chunk.start_time,
                end_time=chunk.end_time,
                native_text=full_text.strip(),
                english_text=english_text,
                language_detected=lang_code,
                confidence=confidence,
            )

        except ImportError:
            logger.info("faster-whisper not available, skipping")
            return None
        except Exception as exc:
            logger.warning(f"faster-whisper error chunk {chunk.chunk_id}: {exc}")
            return None

    # ── Sarvam saarika:v2.5 ───────────────────────────────────────────────────

    @retry(
        retry=retry_if_exception(lambda e: isinstance(e, (httpx.TimeoutException, httpx.HTTPStatusError)) and
              (not isinstance(e, httpx.HTTPStatusError) or e.response.status_code in (429, 503, 502))),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        reraise=True,
    )
    def _transcribe_sarvam(self, chunk: AudioChunk, speaker: Speaker) -> Optional[Utterance]:
        """Sarvam AI saarika:v2.5 — good for South Indian languages."""
        if not settings.sarvam_api_key:
            return None
        chunk_path = Path(chunk.file_path)
        if not chunk_path.exists():
            return None

        try:
            with open(chunk_path, "rb") as f:
                audio_bytes = f.read()

            response = httpx.post(
                SARVAM_STT_URL,
                headers={"api-subscription-key": settings.sarvam_api_key},
                files={"file": (chunk_path.name, audio_bytes, "audio/wav")},
                data={"model": "saarika:v2.5", "language_code": "unknown"},
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()

            native_text   = (data.get("transcript", "") or "").strip()
            language_code = data.get("language_code", "en-IN")
            confidence    = float(data.get("confidence", 1.0))

            if not native_text or _is_hallucination(native_text):
                return None

            english_text = (
                native_text
                if language_code.startswith("en")
                else self._translate_to_english(native_text, language_code)
            )

            return Utterance(
                utterance_id=str(uuid.uuid4()),
                speaker=speaker,
                start_time=chunk.start_time,
                end_time=chunk.end_time,
                native_text=native_text,
                english_text=english_text,
                language_detected=language_code,
                confidence=confidence,
            )
        except Exception as exc:
            logger.debug(f"Sarvam error chunk {chunk.chunk_id}: {exc}")
            return None

    # ── openai-whisper local fallback ─────────────────────────────────────────

    def _transcribe_openai_whisper_local(
        self, chunk: AudioChunk, speaker: Speaker,
    ) -> Optional[Utterance]:
        """Local openai-whisper — last resort fallback."""
        try:
            if self._ow_model is None:
                import whisper as _whisper
                size = getattr(settings, "whisper_model_size", "medium")
                logger.info(f"Loading openai-whisper {size} (fallback)…")
                self._ow_model = _whisper.load_model(size)

            result    = self._ow_model.transcribe(chunk.file_path, fp16=False, task="transcribe")
            text      = (result.get("text", "") or "").strip()
            lang_code = result.get("language", "en")

            if not text or _is_hallucination(text):
                return None

            lc = f"{lang_code}-IN" if "-" not in lang_code else lang_code
            english_text = (
                text if lang_code == "en"
                else self._translate_to_english(text, lc)
            )

            return Utterance(
                utterance_id=str(uuid.uuid4()),
                speaker=speaker,
                start_time=chunk.start_time,
                end_time=chunk.end_time,
                native_text=text,
                english_text=english_text,
                language_detected=lc,
                confidence=0.75,
            )
        except Exception as exc:
            logger.warning(f"openai-whisper error chunk {chunk.chunk_id}: {exc}")
            return None

    # ── Translation ───────────────────────────────────────────────────────────

    def _translate_to_english(self, text: str, source_lang: str) -> str:
        if not text.strip():
            return text
        if settings.openai_api_key:
            return self._translate_openai(text, source_lang)
        if settings.sarvam_api_key:
            return self._translate_sarvam(text, source_lang)
        logger.warning("No translation key configured — returning native text")
        return text

    def _translate_openai(self, text: str, source_lang: str) -> str:
        try:
            from openai import OpenAI
            client     = OpenAI(api_key=settings.openai_api_key)
            lang_label = SARVAM_LANGUAGES.get(source_lang, source_lang)
            response   = client.chat.completions.create(
                model=settings.translation_model,
                messages=[
                    {"role": "system", "content": (
                        "You are a professional translator specialising in Indian languages. "
                        "Translate the following text to English. "
                        "Preserve meaning, tone, and sales/edtech context. "
                        "Keep code-switched English words as-is. "
                        "Return ONLY the translated text — no explanation."
                    )},
                    {"role": "user", "content": f"Source language: {lang_label}\n\nText:\n{text}"},
                ],
                temperature=0.1,
                max_tokens=600,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            logger.warning(f"OpenAI translation failed: {exc}")
            if settings.sarvam_api_key:
                return self._translate_sarvam(text, source_lang)
            return text

    def _translate_sarvam(self, text: str, source_lang: str) -> str:
        try:
            src      = source_lang if "-" in source_lang else f"{source_lang}-IN"
            response = httpx.post(
                SARVAM_TRANS_URL,
                headers={"api-subscription-key": settings.sarvam_api_key,
                         "Content-Type": "application/json"},
                json={
                    "input": text[:2000],
                    "source_language_code": src,
                    "target_language_code": "en-IN",
                    "speaker_gender": "Male",
                    "mode": "formal",
                    "model": "mayura:v1",
                    "enable_preprocessing": False,
                },
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json().get("translated_text", text)
        except Exception as exc:
            logger.warning(f"Sarvam translation failed: {exc}")
            return text
