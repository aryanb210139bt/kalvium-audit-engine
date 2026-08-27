"""
utils/word_normalizer.py
Corrects common STT mis-transcriptions before evaluation.

Mappings are loaded from data/word_mappings.json (user-editable via UI).
Each entry: { "wrong": "Calvium", "correct": "Kalvium", "enabled": true }

Matching is:
  - Case-insensitive (Calvium / calvium / CALVIUM all match)
  - Whole-word only (word boundaries, so "Calvium" won't touch "Calviumx")
  - Replacement preserves the case pattern of the correct word (not the wrong word)
"""
from __future__ import annotations
import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_MAPPINGS_PATH = Path("data/word_mappings.json")
_R2_KEY = "config/word_mappings.json"

# ── Built-in defaults (shipped with the system) ───────────────────────────────
_DEFAULT_MAPPINGS: list[dict] = [
    {"wrong": "Calvium",     "correct": "Kalvium",     "enabled": True, "builtin": True},
    {"wrong": "Kalviyam",    "correct": "Kalvium",     "enabled": True, "builtin": True},
    {"wrong": "Calvin",      "correct": "Kalvium",     "enabled": True, "builtin": True},
    {"wrong": "Calvary",     "correct": "Kalvium",     "enabled": True, "builtin": True},
    {"wrong": "Kalviam",     "correct": "Kalvium",     "enabled": True, "builtin": True},
    {"wrong": "Kalviyum",    "correct": "Kalvium",     "enabled": True, "builtin": True},
    {"wrong": "Calviem",     "correct": "Kalvium",     "enabled": True, "builtin": True},
    {"wrong": "kalviem.com", "correct": "Kalvium.com", "enabled": True, "builtin": True},
    {"wrong": "kalvium.com", "correct": "Kalvium.com", "enabled": True, "builtin": True},  # lowercase domain
    {"wrong": "Kinet",       "correct": "KNET",        "enabled": True, "builtin": True},
    {"wrong": "K-net",       "correct": "KNET",        "enabled": True, "builtin": True},
    {"wrong": "K net",       "correct": "KNET",        "enabled": True, "builtin": True},
    {"wrong": "parent",      "correct": "parents",     "enabled": True, "builtin": True},
]


def load_mappings() -> list[dict]:
    """Return merged list of built-in + user-defined mappings."""
    from storage.persistent_file import sync_from_r2_if_missing
    sync_from_r2_if_missing(_MAPPINGS_PATH, _R2_KEY)

    user: list[dict] = []
    if _MAPPINGS_PATH.exists():
        try:
            data = json.loads(_MAPPINGS_PATH.read_text())
            user = data if isinstance(data, list) else data.get("mappings", [])
        except Exception as e:
            logger.warning(f"Could not load word_mappings.json: {e}")

    # Merge: user mappings take precedence; identify by wrong word (case-insensitive)
    user_keys = {m["wrong"].lower() for m in user}
    merged = [m for m in _DEFAULT_MAPPINGS if m["wrong"].lower() not in user_keys]
    merged.extend(user)
    return merged


def save_mappings(mappings: list[dict]):
    """Persist user-defined mappings (builtin ones are never written to disk)."""
    _MAPPINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Only save non-builtin entries to disk
    to_save = [m for m in mappings if not m.get("builtin")]
    _MAPPINGS_PATH.write_text(json.dumps(to_save, indent=2, ensure_ascii=False))

    from storage.persistent_file import sync_to_r2
    sync_to_r2(_MAPPINGS_PATH, _R2_KEY, content_type="application/json")


# Pre-compile patterns once per process (rebuilt when mappings change)
_compiled: list[tuple[re.Pattern, str]] = []
_compiled_from: list[dict] = []


def _build_patterns(mappings: list[dict]) -> list[tuple[re.Pattern, str]]:
    patterns = []
    for m in mappings:
        if not m.get("enabled", True):
            continue
        wrong   = re.escape(m["wrong"])
        correct = m["correct"]
        try:
            pat = re.compile(rf'\b{wrong}\b', re.IGNORECASE)
            patterns.append((pat, correct))
        except re.error as e:
            logger.warning(f"Bad pattern for '{m['wrong']}': {e}")
    return patterns


def _get_patterns() -> list[tuple[re.Pattern, str]]:
    global _compiled, _compiled_from
    current = load_mappings()
    if current != _compiled_from:
        _compiled      = _build_patterns(current)
        _compiled_from = current
        logger.debug(f"Word normalizer: compiled {len(_compiled)} patterns")
    return _compiled


def normalize(text: str) -> str:
    """Apply all enabled mappings to a piece of text. Returns corrected text."""
    if not text:
        return text
    for pat, replacement in _get_patterns():
        text = pat.sub(replacement, text)
    return text


def normalize_utterances(utterances: list) -> list:
    """
    In-place normalise the english_text (and native_text) of a list of
    Utterance objects or dicts.  Returns the same list for convenience.
    """
    patterns = _get_patterns()
    if not patterns:
        return utterances

    fixed = 0
    for u in utterances:
        if hasattr(u, "english_text"):
            before = u.english_text or ""
            after  = normalize(before)
            if before != after:
                u.english_text = after
                fixed += 1
            # Also fix native text (Hinglish etc. may contain brand names)
            if u.native_text:
                u.native_text = normalize(u.native_text)
        elif isinstance(u, dict):
            before = u.get("english_text", "")
            after  = normalize(before)
            if before != after:
                u["english_text"] = after
                fixed += 1
            if u.get("native_text"):
                u["native_text"] = normalize(u["native_text"])

    if fixed:
        logger.info(f"Word normalizer: corrected {fixed} utterances")
    return utterances
