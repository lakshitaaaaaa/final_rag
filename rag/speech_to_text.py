"""
Speech-to-text fallback for videos without YouTube captions.

Downloads audio via yt-dlp, transcribes it with Groq's free Whisper API
(the same GROQ_API_KEY already used for the LLM), and returns segments in
the same {"text", "start", "duration"} shape produced by the caption path
in transcript.py — so chunking.py and everything downstream doesn't need
to know or care which source a transcript came from.

Requires ffmpeg on PATH (used by yt-dlp for audio extraction, and by this
module to split long audio into chunks that fit Groq's free-tier 25MB
per-request limit):
  - Mac:     brew install ffmpeg
  - Linux:   sudo apt install ffmpeg
  - Windows: https://ffmpeg.org/download.html
"""
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List

from . import config


def _check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError(
            "ffmpeg/ffprobe not found on PATH, but are required for the "
            "speech-to-text fallback. Install ffmpeg first — see the "
            "docstring in rag/speech_to_text.py for platform-specific "
            "instructions — then try again."
        )


def download_audio(video_id: str) -> Path:
    """Downloads audio-only as mono mp3 @ 64kbps — small enough to keep
    even long videos under Groq's free-tier size limit once chunked.
    Cached to disk so re-transcribing doesn't re-download."""
    _check_ffmpeg()
    import yt_dlp  # lazy import: only required when this fallback runs

    out_path = config.AUDIO_CACHE_DIR / f"{video_id}.mp3"
    if out_path.exists():
        return out_path

    url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(config.AUDIO_CACHE_DIR / f"{video_id}.%(ext)s"),
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "64"}
        ],
        "postprocessor_args": ["-ac", "1"],  # mono halves file size again
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    if not out_path.exists():
        raise RuntimeError(
            f"Audio download for '{video_id}' completed but the expected "
            f"file {out_path} wasn't produced — the video may be "
            f"unavailable, age-restricted, or region-locked."
        )
    return out_path


def _get_duration_seconds(audio_path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(audio_path)],
        capture_output=True, text=True, check=True,
    )
    return float(json.loads(result.stdout)["format"]["duration"])


def _split_audio(audio_path: Path, chunk_seconds: int) -> List[Path]:
    """Splits into fixed-length chunks if the audio is longer than
    chunk_seconds; otherwise returns the original file untouched."""
    duration = _get_duration_seconds(audio_path)
    if duration <= chunk_seconds:
        return [audio_path]

    chunk_dir = audio_path.parent / f"{audio_path.stem}_chunks"
    chunk_dir.mkdir(exist_ok=True)
    existing = sorted(chunk_dir.glob(f"{audio_path.stem}_*.mp3"))
    if existing:
        return existing

    pattern = str(chunk_dir / f"{audio_path.stem}_%03d.mp3")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(audio_path), "-f", "segment",
         "-segment_time", str(chunk_seconds), "-c", "copy", pattern],
        capture_output=True, check=True,
    )
    return sorted(chunk_dir.glob(f"{audio_path.stem}_*.mp3"))


def _transcribe_chunk(chunk_path: Path) -> List[Dict]:
    from groq import Groq

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set — it's needed for Whisper transcription too.")
    client = Groq(api_key=api_key)

    with open(chunk_path, "rb") as f:
        transcription = client.audio.transcriptions.create(
            file=(chunk_path.name, f.read()),
            model=config.WHISPER_MODEL_NAME,
            response_format="verbose_json",
        )

    segments = getattr(transcription, "segments", None) or []
    result = []
    for seg in segments:
        text = (seg.get("text") if isinstance(seg, dict) else seg.text).strip()
        if not text:
            continue
        start = seg.get("start") if isinstance(seg, dict) else seg.start
        end = seg.get("end") if isinstance(seg, dict) else seg.end
        result.append({"text": text, "start": float(start), "duration": float(end) - float(start)})
    return result


def transcribe_video(video_id: str) -> List[Dict]:
    """Full fallback pipeline: download audio -> split into <=25MB chunks if
    needed -> transcribe each with Groq Whisper -> merge into one
    timestamp-ordered segment list, in the same shape as caption transcripts."""
    audio_path = download_audio(video_id)
    chunk_paths = _split_audio(audio_path, config.AUDIO_CHUNK_SECONDS)

    all_segments: List[Dict] = []
    for i, chunk_path in enumerate(chunk_paths):
        offset = i * config.AUDIO_CHUNK_SECONDS if len(chunk_paths) > 1 else 0.0
        for seg in _transcribe_chunk(chunk_path):
            seg["start"] += offset
            all_segments.append(seg)

    if not all_segments:
        raise RuntimeError(
            f"Whisper transcription for '{video_id}' produced no text — "
            "the audio may be silent, music-only, or in an unsupported format."
        )
    return all_segments
