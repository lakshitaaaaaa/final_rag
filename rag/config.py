"""Central configuration for paths, models, and tunable parameters."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TRANSCRIPT_CACHE_DIR = DATA_DIR / "transcripts"
FAISS_INDEX_DIR = DATA_DIR / "faiss_index"
MANIFEST_PATH = DATA_DIR / "manifest.json"

for d in (TRANSCRIPT_CACHE_DIR, FAISS_INDEX_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Embedding model — runs locally via sentence-transformers, no API key, no cost.
# Multilingual model: puts semantically similar text from *different*
# languages close together in the same embedding space, so a question asked
# in one language can retrieve relevant chunks from a video transcribed in
# a completely different language. Same 384-dim output as the previous
# English-only MiniLM model, but NOT embedding-compatible with it — if you
# switch from the old model, re-index existing videos (see scripts/reindex_all.py).
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# LLM — Groq's free-tier API (OpenAI-compatible), fast Llama models, no cost
# for reasonable usage. Get a free key at https://console.groq.com/keys
GROQ_MODEL_NAME = os.environ.get("GROQ_MODEL_NAME", "llama-3.1-8b-instant")

# Speech-to-text fallback (used only when a video has no YouTube captions).
# Same free Groq account/key as the LLM above.
ENABLE_WHISPER_FALLBACK = os.environ.get("ENABLE_WHISPER_FALLBACK", "true").lower() == "true"
WHISPER_MODEL_NAME = os.environ.get("WHISPER_MODEL_NAME", "whisper-large-v3-turbo")
AUDIO_CACHE_DIR = DATA_DIR / "audio"
AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
# Groq's free tier caps requests at 25MB each. At 64kbps mono mp3, a 20-minute
# chunk is comfortably under that (~9-10MB), leaving headroom.
AUDIO_CHUNK_SECONDS = 1200

# Chunking
CHUNK_SIZE_CHARS = 1000
CHUNK_OVERLAP_CHARS = 200

# Retrieval
TOP_K = 4
# FAISS uses L2 distance by default (lower = more similar). If the best chunk's
# distance is above this, we treat the question as "not covered by the video(s)"
# rather than letting the LLM guess. Tune this if you change the embedding model.
MAX_DISTANCE_FOR_GROUNDED_ANSWER = 1.0
