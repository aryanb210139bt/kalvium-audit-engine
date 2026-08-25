"""
transcription/faster_whisper_stt.py
Faster-Whisper Large-V3 integration for English, Hindi, and Hinglish.

Uses CTranslate2 INT8 quantization — 15–20× real-time on A10G.
Falls back to openai-whisper (base) if faster-whisper is not installed.

Word-level timestamps are enabled for speaker-transcript alignment.
"""
from __future__ import annotations
import logging
import re
import uuid
from pathlib import Path
from typing import Optional

from config.models_multilingual import (
    AudioChunkV2, Language, SpeakerRole, Utterance, WordToken
)
from config.settings import get_settings

logger   = logging.getLogger(__name__)
settings = get_settings()

# Whisper language code for forced decoding (speeds up + reduces hallucination)
LANG_TO_WHISPER: dict[Language, str] = {
    Language.ENGLISH:  "en",
    Language.HINDI:    "hi",
    Language.HINGLISH: "hi",   # Whisper decodes Hinglish under Hindi
    Language.KANNADA:  "kn",
    Language.TAMIL:    "ta",
    Language.TELUGU:   "te",
    Language.MALAYALAM: "ml",
    Language.BENGALI:  "bn",
    Language.MARATHI:  "mr",
    Language.GUJARATI: "gu",
    Language.PUNJABI:  "pa",
}

HALLUCINATION_PHRASES = {
    "thank you for watching",
    "subscribe to our channel",
    "please like and subscribe",
    "www.",
    "subtitles by",
    "transcribed by",
}


class FasterWhisperSTT:
    """
    Transcribes audio chunks using Faster-Whisper (or fallback Whisper).
    Singleton pattern — model is loaded once and reused.
    """

    def __init__(self):
        self._model  = None
        self._engine = "none"
        self._load()

    def _load(self) -> None:
        # Try faster-whisper first
        try:
            from faster_whisper import WhisperModel
            model_size = getattr(settings, "whisper_model_size", "large-v3")
            device     = getattr(settings, "whisper_device", "cpu")
            compute    = "int8" if device == "cpu" else "float16"
            self._model  = WhisperModel(model_size, device=device, compute_type=compute)
            self._engine = "faster-whisper"
            logger.info(f"Faster-Whisper {model_size} loaded ({device}/{compute})")
            return
        except ImportError:
            logger.info("faster-whisper not installed; trying openai-whisper…")
        except Exception as exc:
            logger.warning(f"faster-whisper load failed ({exc}); trying fallback…")

        # Fallback: openai-whisper
        try:
            import whisper
            size = getattr(settings, "whisper_model_size", "base")
            self._model  = whisper.load_model(size)
            self._engine = "openai-whisper"
            logger.info(f"Whisper {size} loaded (fallback)")
        except Exception as exc:
            logger.error(f"Whisper fallback also failed: {exc}")
            self._engine = "none"

    def transcribe_chunk(
        self,
        chunk: AudioChunkV2,
        speaker: SpeakerRole,
        primary_language: Language = Language.UNKNOWN,
    ) -> Optional[Utterance]:
        """Transcribe a single audio chunk. Returns None if empty/hallucination."""
        if self._engine == "none":
            return None

        wav_path = Path(chunk.file_path)
        if not wav_path.exists():
            return None

        whisper_lang = LANG_TO_WHISPER.get(primary_language)

        if self._engine == "faster-whisper":
            return self._transcribe_fw(chunk, speaker, wav_path, whisper_lang)
        return self._transcribe_ow(chunk, speaker, wav_path, whisper_lang)

    # ── Faster-Whisper ─────────────────────────────────────────────────────────

    def _transcribe_fw(
        self,
        chunk: AudioChunkV2,
        speaker: SpeakerRole,
        wav_path: Path,
        language: Optional[str],
    ) -> Optional[Utterance]:
        try:
            segments, info = self._model.transcribe(
                str(wav_path),
                language=language,
                beam_size=5,
                word_timestamps=True,
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 300},
                temperature=0.0,
                no_speech_threshold=0.5,
            )

            words: list[WordToken] = []
            full_text = ""
            total_prob = 0.0
            seg_count = 0

            for seg in segments:
                text = seg.text.strip()
                if _is_hallucination(text):
                    continue
                full_text += (" " if full_text else "") + text
                total_prob += seg.avg_logprob
                seg_count += 1
                if seg.words:
                    for w in seg.words:
                        words.append(WordToken(
                            word=w.word.strip(),
                            start=chunk.start_time + w.start,
                            end=chunk.start_time + w.end,
                            confidence=float(w.probability),
                            speaker=speaker,
                        ))

            if not full_text.strip():
                return None

            avg_logprob = total_prob / max(seg_count, 1)
            confidence  = min(1.0, max(0.0, 1.0 + avg_logprob / 5))  # rough normalize

            detected_lang = _whisper_lang_to_enum(info.language) \
                if hasattr(info, "language") else Language.UNKNOWN

            return Utterance(
                utterance_id=str(uuid.uuid4()),
                speaker=speaker,
                start_time=chunk.start_time,
                end_time=chunk.end_time,
                text=full_text.strip(),
                english_text=full_text.strip(),   # translate in post-processing
                language=detected_lang,
                confidence=round(confidence, 3),
                words=words,
                is_question="?" in full_text,
            )
        except Exception as exc:
            logger.warning(f"Faster-Whisper inference error: {exc}")
            return None

    # ── OpenAI Whisper fallback ────────────────────────────────────────────────

    def _transcribe_ow(
        self,
        chunk: AudioChunkV2,
        speaker: SpeakerRole,
        wav_path: Path,
        language: Optional[str],
    ) -> Optional[Utterance]:
        try:
            import whisper
            try:
                opts = {"language": language, "fp16": False, "word_timestamps": True}
                result = self._model.transcribe(str(wav_path), **opts)
            except Exception:
                opts = {"language": language, "fp16": False}
                result = self._model.transcribe(str(wav_path), **opts)
            text = result.get("text", "").strip()
            if not text or _is_hallucination(text):
                return None

            words: list[WordToken] = []
            for seg in result.get("segments", []):
                for w in seg.get("words", []):
                    words.append(WordToken(
                        word=w["word"].strip(),
                        start=chunk.start_time + w["start"],
                        end=chunk.start_time + w["end"],
                        confidence=float(w.get("probability", 1.0)),
                        speaker=speaker,
                    ))

            detected = _whisper_lang_to_enum(result.get("language", "en"))
            return Utterance(
                utterance_id=str(uuid.uuid4()),
                speaker=speaker,
                start_time=chunk.start_time,
                end_time=chunk.end_time,
                text=text,
                english_text=text,
                language=detected,
                confidence=0.85,
                words=words,
                is_question="?" in text,
            )
        except Exception as exc:
            logger.warning(f"OpenAI Whisper error: {exc}")
            return None


# ── Helpers ────────────────────────────────────────────────────────────────────

def _is_hallucination(text: str) -> bool:
    t = text.strip()
    if not t or len(t) < 4:
        return True
    tl = t.lower()
    for phrase in HALLUCINATION_PHRASES:
        if phrase in tl:
            return True
    # Single char repeated 6+ times (e.g. ◆◆◆◆◆◆, repeated Devanagari chars)
    if re.search(r'(.)\1{5,}', t):
        return True
    # Short sequence repeated 4+ consecutive times
    if re.search(r'(.{2,8})\1{3,}', t):
        return True
    # Only numbers/dots/spaces (e.g. "3. 3. 2. 2.")
    if re.fullmatch(r'[\d\s.,\-]+', t):
        return True
    # Word-level bigram repetition
    words = tl.split()
    if len(words) > 8:
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
        if len(set(bigrams)) / len(bigrams) < 0.45:
            return True
    return False


def _whisper_lang_to_enum(lang_code: str) -> Language:
    mapping = {
        "en": Language.ENGLISH, "hi": Language.HINDI,
        "kn": Language.KANNADA, "ta": Language.TAMIL,
        "te": Language.TELUGU,  "ml": Language.MALAYALAM,
        "bn": Language.BENGALI, "mr": Language.MARATHI,
        "gu": Language.GUJARATI,"pa": Language.PUNJABI,
    }
    return mapping.get(lang_code, Language.UNKNOWN)
