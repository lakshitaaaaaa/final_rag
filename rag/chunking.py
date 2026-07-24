"""
Group raw transcript segments into overlapping text chunks while keeping
track of the start timestamp of each chunk, so retrieved answers can be
linked back to the exact moment in the video.
"""
from typing import List, Dict

from . import config


def chunk_transcript(
    segments: List[Dict],
    chunk_size: int = config.CHUNK_SIZE_CHARS,
    chunk_overlap: int = config.CHUNK_OVERLAP_CHARS,
) -> List[Dict]:
    """
    segments: [{"text": str, "start": float, "duration": float}, ...]
    Returns: [{"text": str, "start": float}, ...] chunks, each carrying the
    start time (in seconds) of the first transcript segment it contains.
    """
    if not segments:
        return []

    chunks: List[Dict] = []
    buf_texts: List[str] = []
    buf_start = segments[0]["start"]
    buf_len = 0
    i = 0
    n = len(segments)

    while i < n:
        seg = segments[i]
        if not buf_texts:
            buf_start = seg["start"]
        buf_texts.append(seg["text"])
        buf_len += len(seg["text"]) + 1

        is_last = i == n - 1
        if buf_len >= chunk_size or is_last:
            chunk_text = " ".join(buf_texts).strip()
            if chunk_text:
                chunks.append({"text": chunk_text, "start": buf_start})

            if is_last:
                break

            # Build overlap: walk backwards from current position, collecting
            # segments until we've covered ~chunk_overlap characters, so the
            # next chunk starts with some shared context.
            overlap_texts: List[str] = []
            overlap_len = 0
            j = i
            while j >= 0 and overlap_len < chunk_overlap:
                overlap_texts.insert(0, segments[j]["text"])
                overlap_len += len(segments[j]["text"]) + 1
                j -= 1

            buf_texts = overlap_texts
            buf_len = overlap_len
            buf_start = segments[j + 1]["start"] if j + 1 <= i else seg["start"]

        i += 1

    return chunks
