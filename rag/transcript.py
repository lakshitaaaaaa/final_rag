"""Fetch and cache YouTube transcripts, preserving per-segment timestamps."""
import json
import re
from typing import List, Dict, Optional

import requests
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)

from . import config


def extract_video_id(url_or_id: str) -> str:
    """Accepts a full YouTube URL (watch, youtu.be, shorts) or a bare video ID."""
    url_or_id = url_or_id.strip()
    patterns = [
        r"(?:v=|/videos/|embed/|youtu\.be/|/shorts/|/live/)([0-9A-Za-z_-]{11})",
    ]
    for pat in patterns:
        m = re.search(pat, url_or_id)
        if m:
            return m.group(1)
    # Already looks like a bare 11-char video ID
    if re.fullmatch(r"[0-9A-Za-z_-]{11}", url_or_id):
        return url_or_id
    raise ValueError(f"Could not extract a YouTube video ID from: {url_or_id}")


def fetch_video_title(video_id: str) -> str:
    """Best-effort title lookup via YouTube's free oEmbed endpoint (no API key)."""
    try:
        resp = requests.get(
            "https://www.youtube.com/oembed",
            params={"url": f"https://www.youtube.com/watch?v={video_id}", "format": "json"},
            timeout=8,
        )
        if resp.ok:
            return resp.json().get("title", video_id)
    except requests.RequestException:
        pass
    return video_id


def _cache_path(video_id: str):
    return config.TRANSCRIPT_CACHE_DIR / f"{video_id}.json"


def get_transcript_source(video_id: str) -> Optional[str]:
    """Returns 'youtube_captions' or 'whisper_fallback' for an already-cached
    video, or None if it hasn't been fetched yet."""
    cache_file = _cache_path(video_id)
    if not cache_file.exists():
        return None
    data = json.loads(cache_file.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return "youtube_captions"  # legacy cache files predate the source field
    return data.get("source", "youtube_captions")


def fetch_transcript(video_id: str, languages: Optional[List[str]] = None) -> List[Dict]:
    """
    Returns a list of {"text": str, "start": float, "duration": float} segments.
    Uses a local JSON cache so re-indexing the same video doesn't re-hit the API.

    `languages` is a preference-ordered list of caption language codes to try
    first (YouTube captions, when available, are free and instant). It
    defaults to a broad set of common languages rather than English-only,
    since this app supports multilingual video content. Any video whose
    captions aren't in this list — or that has no captions at all — falls
    through to the Whisper fallback below, which is language-agnostic, so
    caption language coverage here is a speed/cost optimization, not a hard
    requirement.

    If the video has no YouTube captions and config.ENABLE_WHISPER_FALLBACK is
    True, falls back to downloading the audio and transcribing it with Groq's
    free Whisper API (see rag/speech_to_text.py). The returned shape is
    identical either way, so callers don't need to know which path was used;
    use get_transcript_source() to check afterward.
    """
    languages = languages or [
        "en", "en-US", "en-GB", "hi", "es", "fr", "de", "pt", "ru", "ja",
        "ko", "zh-Hans", "zh-Hant", "ar", "bn", "ta", "te", "mr", "ur", "it",
    ]
    cache_file = _cache_path(video_id)
    if cache_file.exists():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        # Legacy cache files were a bare list of segments; support both.
        return cached["segments"] if isinstance(cached, dict) else cached

    try:
        api = YouTubeTranscriptApi()
        fetched = api.fetch(video_id, languages=languages)
        segments = [
            {"text": s.text, "start": s.start, "duration": s.duration}
            for s in fetched
        ]
        source = "youtube_captions"
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
        if not config.ENABLE_WHISPER_FALLBACK:
            raise RuntimeError(
                f"No usable transcript for video '{video_id}': {e}. "
                "Set ENABLE_WHISPER_FALLBACK=true to transcribe the audio "
                "with Whisper instead of relying on YouTube captions."
            ) from e
        try:
            from . import speech_to_text
            segments = speech_to_text.transcribe_video(video_id)
            source = "whisper_fallback"
        except Exception as whisper_error:
            raise RuntimeError(
                f"No YouTube captions for video '{video_id}' ({e}), and the "
                f"Whisper fallback also failed: {whisper_error}"
            ) from whisper_error

    cache_file.write_text(
        json.dumps({"source": source, "segments": segments}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return segments


def format_timestamp(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
