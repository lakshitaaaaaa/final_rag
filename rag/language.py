"""
Local, free language detection — used to answer questions in whatever
language they were asked in, regardless of the language of the video's
transcript. Runs via `langdetect` (pure Python, no API call, no cost).
"""
from typing import Optional, Tuple

from langdetect import DetectorFactory, LangDetectException, detect_langs

# Makes detection deterministic (langdetect is otherwise seeded randomly,
# which can flip results for short/ambiguous text between runs).
DetectorFactory.seed = 0

# Common language codes -> human-readable names, used to make the LLM prompt
# clearer than a bare ISO code (e.g. "Hindi" instead of "hi"). Not
# exhaustive — langdetect supports ~55 languages; unlisted codes just get
# passed through as-is, which Llama models generally still understand.
LANGUAGE_NAMES = {
    "en": "English", "hi": "Hindi", "es": "Spanish", "fr": "French",
    "de": "German", "pt": "Portuguese", "ru": "Russian", "ja": "Japanese",
    "ko": "Korean", "zh-cn": "Chinese", "zh-tw": "Chinese", "ar": "Arabic",
    "bn": "Bengali", "ta": "Tamil", "te": "Telugu", "mr": "Marathi",
    "ur": "Urdu", "it": "Italian", "nl": "Dutch", "tr": "Turkish",
    "vi": "Vietnamese", "th": "Thai", "id": "Indonesian", "pl": "Polish",
    "gu": "Gujarati", "kn": "Kannada", "ml": "Malayalam", "pa": "Punjabi",
}

# Below this length, detection is too unreliable to attempt at all (single
# words, "hi", "ok", etc).
MIN_CHARS_FOR_DETECTION = 12
# Even above that length, only trust the result if langdetect itself is
# reasonably confident — short/common words in one language can otherwise
# get misread as a different language (e.g. "summarize" as Italian).
MIN_CONFIDENCE = 0.85


def detect_language(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Returns (iso_code, human_name), or (None, None) if detection isn't
    reliable enough to trust (too short, ambiguous, or low-confidence)."""
    if not text or len(text.strip()) < MIN_CHARS_FOR_DETECTION:
        return None, None
    try:
        candidates = detect_langs(text)
    except LangDetectException:
        return None, None
    if not candidates or candidates[0].prob < MIN_CONFIDENCE:
        return None, None
    code = candidates[0].lang
    return code, LANGUAGE_NAMES.get(code, code)


def target_language_instruction(text: str) -> str:
    """A short phrase to slot into the LLM prompt telling it what language
    to answer in. Falls back to an implicit instruction (still usually
    correct — Llama models are decent at inferring the question's language
    on their own) when detection isn't confident enough to name one."""
    _, name = detect_language(text)
    if name:
        return f"in {name} (the language of the question)"
    return "in the same language the question was asked in"
