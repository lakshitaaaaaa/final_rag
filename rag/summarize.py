"""
Whole-video summarization.

Retrieval-based QA (rag/qa_chain.py) is the wrong tool for "summarize this
video": it only pulls the top-k chunks most similar to the word
"summarize", not the whole transcript. This module instead does a
map-reduce pass over every chunk of the transcript:

  1. Map:    summarize each chunk independently
  2. Reduce: combine those partial summaries into one coherent summary

This reuses the transcript cache (no re-download) and the existing
chunking function, just with a larger chunk size — summarization needs
fewer, bigger pieces than retrieval does, to keep the number of LLM calls
(and therefore latency/rate-limit usage) reasonable even for long videos.
"""
import re
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from . import transcript, chunking, qa_chain

_SUMMARY_PATTERNS = [
    r"\bsummari[sz]e\b", r"\bsummary\b", r"\btl;?dr\b",
    r"\bgive me an overview\b", r"\bwhat is this video about\b",
    r"\bwhat.?s this video about\b", r"\brecap\b",
]
_SUMMARY_RE = re.compile("|".join(_SUMMARY_PATTERNS), re.IGNORECASE)


def is_summary_request(text: str) -> bool:
    """Heuristic keyword match — not a full intent classifier, but catches
    the common phrasings without adding an extra LLM call just to detect
    intent. False negatives fall through to normal retrieval-based QA,
    which still gives a reasonable (if less complete) answer."""
    return bool(_SUMMARY_RE.search(text or ""))

MAP_PROMPT = ChatPromptTemplate.from_template(
    """Summarize the key points of this transcript excerpt in 2-4 concise
sentences. Do not add any information not present in the excerpt.

Excerpt:
{text}

Summary:"""
)

REDUCE_PROMPT = ChatPromptTemplate.from_template(
    """You are given partial summaries of consecutive sections of a video
transcript, in order. Combine them into one coherent, well-organized
summary of the whole video. Remove redundancy between sections, but don't
drop distinct points. Do not add any information not present below.

Partial summaries:
{partial_summaries}

Write the final summary {target_language}, as flowing prose (not a list of
the sections), in about 150-250 words:"""
)

# Cap the number of map calls for very long videos: rather than growing
# without bound (and burning through the free API's rate limits), the
# per-chunk size grows so total map calls stay roughly constant.
TARGET_MAP_CALLS = 12
MIN_MAP_CHUNK_CHARS = 3000
MAP_CHUNK_OVERLAP = 200


def _pick_map_chunk_size(total_chars: int) -> int:
    return max(MIN_MAP_CHUNK_CHARS, total_chars // TARGET_MAP_CALLS)


def summarize_video(video_id: str, target_language: str = "in English") -> dict:
    """Returns {"summary": str, "num_sections": int}. target_language should
    be a phrase like the output of rag.language.target_language_instruction,
    e.g. "in Hindi (the language of the question)"."""
    segments = transcript.fetch_transcript(video_id)  # cached, no re-fetch
    if not segments:
        return {"summary": "No transcript available for this video.", "num_sections": 0}

    total_chars = sum(len(s["text"]) for s in segments)
    map_chunk_size = _pick_map_chunk_size(total_chars)
    map_chunks = chunking.chunk_transcript(segments, map_chunk_size, MAP_CHUNK_OVERLAP)

    llm = qa_chain.get_llm()
    map_chain = MAP_PROMPT | llm | StrOutputParser()
    reduce_chain = REDUCE_PROMPT | llm | StrOutputParser()

    partial_summaries: List[str] = [
        map_chain.invoke({"text": c["text"]}) for c in map_chunks
    ]

    if len(partial_summaries) == 1:
        # Short video — one chunk's summary IS the summary, no need to
        # reduce (and reduce prompt tends to pad short input unnecessarily).
        final_summary = partial_summaries[0]
    else:
        combined = "\n\n".join(f"[Section {i+1}] {s}" for i, s in enumerate(partial_summaries))
        final_summary = reduce_chain.invoke({
            "partial_summaries": combined,
            "target_language": target_language,
        })

    return {"summary": final_summary, "num_sections": len(map_chunks)}
