"""
transcription/asr_router.py
Routes each audio chunk to the appropriate ASR model based on language.

Routing table:
  English / Hindi / Hinglish / Unknown → Faster-Whisper Large-V3
  Tamil / Kannada / Telugu / Malayalam / Bengali / Marathi → Sarvam Saarika v2
  Heavy code-switch (density > 0.4) → Faster-Whisper (handles mixed better)

Falls back to Sarvam API → Whisper base in that order.
"""
from __future__ import annotations
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from config.models_multilingual import (
    AudioChunkV2, Language, LanguageMap, SpeakerRole, Utterance, WordToken
)
from config.settings import get_settings

logger   = logging.getLogger(__name__)
settings = get_settings()

# Languages that Sarvam Saarika v2 handles better than Whisper
SARVAM_PREFERRED = {
    Language.TAMIL, Language.KANNADA, Language.TELUGU,
    Language.MALAYALAM, Language.BENGALI, Language.MARATHI,
    Language.GUJARATI, Language.PUNJABI,
}

MAX_WORKERS = 6   # parallel ASR workers


class ASRRouter:
    """
    Orchestrates parallel ASR across chunks with per-chunk model routing.
    """

    def __init__(self):
        from transcription.faster_whisper_stt import FasterWhisperSTT
        from transcription.stt import SpeechToText as SarvamSTT
        self._fw    = FasterWhisperSTT()
        self._sarvam = SarvamSTT()

    def transcribe_all(
        self,
        chunks:       list[AudioChunkV2],
        language_map: LanguageMap,
        speaker_map:  dict[int, SpeakerRole],   # chunk_id → speaker
    ) -> list[Utterance]:
        """
        Transcribe all chunks in parallel, routing to best model per chunk.
        Returns utterances sorted by start_time.
        """
        args = [
            (chunk, language_map, speaker_map.get(chunk.chunk_id, SpeakerRole.UNKNOWN))
            for chunk in chunks
        ]

        utterances: list[Utterance] = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {
                pool.submit(self._transcribe_one, *a): a[0].chunk_id
                for a in args
            }
            for future in as_completed(futures):
                chunk_id = futures[future]
                try:
                    utt = future.result()
                    if utt and utt.text.strip():
                        utterances.append(utt)
                except Exception as exc:
                    logger.warning(f"ASR failed for chunk {chunk_id}: {exc}")

        utterances.sort(key=lambda u: u.start_time)
        logger.info(f"ASR complete: {len(utterances)}/{len(chunks)} non-empty utterances")
        return utterances

    def _transcribe_one(
        self,
        chunk:        AudioChunkV2,
        language_map: LanguageMap,
        speaker:      SpeakerRole,
    ) -> Optional[Utterance]:
        """Choose model and transcribe a single chunk."""
        use_sarvam = self._should_use_sarvam(chunk, language_map)

        if use_sarvam and settings.sarvam_api_key:
            utt = self._run_sarvam(chunk, speaker)
            if utt:
                return utt
            logger.debug(f"Chunk {chunk.chunk_id}: Sarvam failed, falling back to Whisper")

        # Faster-Whisper (primary or fallback)
        return self._run_faster_whisper(chunk, speaker, language_map)

    def _should_use_sarvam(self, chunk: AudioChunkV2, lm: LanguageMap) -> bool:
        """Use Sarvam for pure Indic chunks; Whisper for everything else."""
        if lm.code_switch_density > 0.4:
            return False   # heavy code-switching → Whisper handles better
        chunk_lang = self._chunk_language(chunk, lm)
        return chunk_lang in SARVAM_PREFERRED

    def _chunk_language(self, chunk: AudioChunkV2, lm: LanguageMap) -> Language:
        """Find language for this chunk's time window."""
        mid = (chunk.start_time + chunk.end_time) / 2
        for seg in lm.segments:
            if seg.start <= mid <= seg.end:
                return seg.language
        return lm.primary_language

    def _run_sarvam(self, chunk: AudioChunkV2, speaker: SpeakerRole) -> Optional[Utterance]:
        try:
            import httpx
            chunk_path = Path(chunk.file_path)
            with open(chunk_path, "rb") as f:
                audio_bytes = f.read()

            response = httpx.post(
                "https://api.sarvam.ai/speech-to-text",
                headers={"api-subscription-key": settings.sarvam_api_key},
                files={"file": (chunk_path.name, audio_bytes, "audio/wav")},
                data={"model": "saarika:v2.5", "language_code": "unknown"},
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()

            native_text = data.get("transcript", "").strip()
            if not native_text:
                return None

            lang_code = data.get("language_code", "en-IN")
            lang = _sarvam_lang_code(lang_code)
            confidence = float(data.get("confidence", 1.0))

            english_text = native_text if lang == Language.ENGLISH else \
                self._translate(native_text, lang)

            return Utterance(
                utterance_id=str(uuid.uuid4()),
                speaker=speaker,
                start_time=chunk.start_time,
                end_time=chunk.end_time,
                text=native_text,
                english_text=english_text,
                language=lang,
                confidence=confidence,
            )
        except Exception as exc:
            logger.debug(f"Sarvam API error: {exc}")
            return None

    def _run_faster_whisper(
        self,
        chunk: AudioChunkV2,
        speaker: SpeakerRole,
        lm: LanguageMap,
    ) -> Optional[Utterance]:
        try:
            return self._fw.transcribe_chunk(chunk, speaker, lm.primary_language)
        except Exception as exc:
            logger.warning(f"Faster-Whisper failed for chunk {chunk.chunk_id}: {exc}")
            return None

    def _translate(self, text: str, from_lang: Language) -> str:
        """Best-effort translation via OpenAI if key available."""
        if not settings.openai_api_key or from_lang == Language.ENGLISH:
            return text
        try:
            from openai import OpenAI
            client = OpenAI(api_key=settings.openai_api_key)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "system",
                    "content": "Translate the following Indian language text to English. "
                               "Preserve meaning, tone, and sales context. "
                               "Return ONLY the translation, no explanations."
                }, {
                    "role": "user",
                    "content": text,
                }],
                max_tokens=500,
                temperature=0,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return text   # return original if translation fails


def _sarvam_lang_code(code: str) -> Language:
    mapping = {
        "hi-IN": Language.HINDI,
        "ta-IN": Language.TAMIL,
        "te-IN": Language.TELUGU,
        "kn-IN": Language.KANNADA,
        "ml-IN": Language.MALAYALAM,
        "bn-IN": Language.BENGALI,
        "mr-IN": Language.MARATHI,
        "gu-IN": Language.GUJARATI,
        "pa-IN": Language.PUNJABI,
        "en-IN": Language.ENGLISH,
    }
    return mapping.get(code, Language.UNKNOWN)
