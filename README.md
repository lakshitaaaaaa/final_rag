# YouTube RAG Assistant

A Retrieval-Augmented Generation (RAG) app that answers questions grounded in
the transcripts of one or more YouTube videos, with answers linked back to
the exact timestamp they came from.

This is an upgrade of the original single-video Colab notebook into a
persistent, multi-video, deployable app.

## What's new vs. the notebook version

| | Notebook version | This version |
|---|---|---|
| Interface | Jupyter/Colab cells | Streamlit chat UI |
| Scope | One video per run | Index many videos, search one or all |
| Vector store | Rebuilt every run, in-memory | Persisted to disk, reused across sessions |
| Answers | Plain text | Linked to `youtu.be/...&t=123s` timestamps |
| Ungrounded questions | LLM may guess | Similarity-distance guard refuses to answer if nothing relevant was retrieved |
| LLM | `ChatOpenAI(model="llama3")` (mismatched client/model) | `ChatGroq` — genuinely free-tier, OpenAI-compatible |
| Evaluation | None | `eval/` harness: LLM-judged correctness + faithfulness, retrieval localization, out-of-scope refusal rate, config comparison |
| Caption-less videos | Fails outright | Falls back to Whisper transcription (via Groq's free API) automatically |
| Language support | English only | Multilingual indexing + true cross-lingual Q&A (ask in any language, get an answer in that language, regardless of the video's language) |
| Summarization | Not possible (retrieval only returns top-k chunks) | Dedicated map-reduce summarizer — button or just ask "summarize this video" in chat |

## Architecture

```
YouTube URL
    |
    v
Transcript Fetcher (youtube-transcript-api) --cached--> data/transcripts/<id>.json
    |
    +-- captions available -----------------> segments
    |
    +-- no captions (fallback, if enabled):
             yt-dlp audio download --cached--> data/audio/<id>.mp3
                     |
                     v
             ffmpeg split into <=20 min chunks (Groq free-tier 25MB cap)
                     |
                     v
             Groq Whisper API (whisper-large-v3-turbo) --> segments
    |
    v
Timestamp-preserving Chunker (chunk_size=1000, overlap=200)
    |
    v
HuggingFace Embeddings (all-MiniLM-L6-v2, local, free)
    |
    v
FAISS index  --persisted--> data/faiss_index/<video_id>/  and  data/faiss_index/_all/
    |
    v
Retriever (similarity search, k=4) --> distance guard
    |
    v
Prompt assembly (numbered, timestamped excerpts)
    |
    v
LLM (Groq, Llama 3.1 8B Instant) --> grounded answer + cited sources
```

## Project structure

```
youtube-rag-assistant/
├── app.py                  # Streamlit UI
├── rag/
│   ├── config.py            # paths, model names, chunk/retrieval params
│   ├── transcript.py        # fetch + cache transcripts, video ID/title helpers
│   ├── speech_to_text.py    # Whisper fallback (audio download, chunking, transcription)
│   ├── language.py          # local language detection for cross-lingual answers
│   ├── chunking.py          # timestamp-preserving chunking
│   ├── vectorstore.py       # FAISS build/persist/load (per-video + combined)
│   ├── qa_chain.py          # retrieval, grounding guard, prompt, LLM call
│   ├── summarize.py         # map-reduce whole-video summarization
│   └── ingest.py            # ties the above into one ingest_video() call
├── scripts/
│   └── reindex_all.py       # rebuild all indexes after an embedding-model change
├── eval/
│   ├── eval_set.example.json
│   ├── metrics.py
│   ├── run_eval.py
│   └── compare.py
├── data/
│   ├── transcripts/         # cached raw transcripts (json)
│   ├── audio/                # cached downloaded audio (Whisper fallback only)
│   ├── faiss_index/         # persisted vector indexes
│   └── manifest.json        # which videos have been indexed
├── requirements.txt
└── .env.example
```

## Prerequisites

- Python 3.9–3.12
- **ffmpeg** on PATH — required for the Whisper fallback (audio extraction
  and chunking). Not needed if you only ever index videos that already have
  YouTube captions.
  - Mac: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg`
  - Windows: https://ffmpeg.org/download.html

## Setup

1. **Clone and install**
   ```bash
   git clone https://github.com/<your-username>/youtube-rag-assistant.git
   cd youtube-rag-assistant
   pip install -r requirements.txt
   ```

2. **Get a free Groq API key** at https://console.groq.com/keys (no cost —
   Groq's free tier is generous and fast). Copy `.env.example` to `.env` and
   paste your key, or just paste it into the sidebar at runtime.

3. **Run**
   ```bash
   streamlit run app.py
   ```

4. In the sidebar, paste a YouTube URL and click **Index video**. Once
   indexed, ask questions in the chat box. Index more videos to build up a
   searchable library and switch the "Search scope" dropdown between a
   single video or all of them.

## Evaluation harness

Running the app manually tells you it *works*; it doesn't tell you it works
*well*, or that your design choices (chunk size, k, distance threshold) are
good ones. The `eval/` package addresses that:

```
eval/
├── eval_set.example.json   # template — copy to eval_set.json and fill in
├── metrics.py               # LLM-as-judge: faithfulness + correctness scoring
├── run_eval.py               # runs one config through the eval set
├── compare.py                 # runs several configs, prints a comparison table
└── results/                   # output JSON + per-config isolated indexes
```

**What it measures, per question:**
- **Correctness** (1-5, LLM-judged against a reference answer you write)
- **Faithfulness** (1-5, LLM-judged: is the answer actually supported by the
  retrieved context, or did the model add unsupported claims?)
- **Retrieval localization hit** — if you supply an expected timestamp, did
  any retrieved chunk actually come from near that point in the video?
- **Out-of-scope refusal rate** — for questions the video doesn't cover, did
  the grounding guard correctly say "not covered" instead of hallucinating?

**Setup:**
1. Index 2-3 videos in the app first.
2. `cp eval/eval_set.example.json eval/eval_set.json` and fill in real
   questions, your own reference answers, and a few deliberately
   out-of-scope questions (aim for 20-30 total).
3. Run a single config:
   ```bash
   python -m eval.run_eval --label baseline
   ```
4. Compare design choices (edit the `CONFIGS` list at the top of
   `eval/compare.py` first):
   ```bash
   python -m eval.compare
   ```
   This prints a markdown table you can paste straight into your project
   report, e.g.:

   | label | chunk_size | k | avg_correctness | avg_faithfulness | retrieval_localization_hit_rate | out_of_scope_correct_refusal_rate |
   |---|---|---|---|---|---|---|
   | chunk500_overlap100_k4 | 500 | 4 | 4.2 | 4.6 | 0.8 | 1.0 |
   | chunk1000_overlap200_k4 | 1000 | 4 | 4.5 | 4.7 | 0.7 | 1.0 |
   | chunk1000_overlap200_k2 | 1000 | 2 | 3.9 | 4.8 | 0.6 | 1.0 |

   (numbers above are illustrative — run it on your own videos to get real ones)

Each config indexes into its own disposable folder under
`eval/results/<label>/indexes/`, so experiments never touch your live
`data/faiss_index/` used by the app.

**Note on cost/rate limits:** each question makes 1 LLM call for the answer
plus up to 2 more for judging (correctness + faithfulness), so a 30-question
eval run makes ~90 Groq calls. This comfortably fits in Groq's free tier for
occasional runs, but avoid running `compare.py` across many configs back to
back if you're rate-limited — add a short `time.sleep()` in `run_eval.py`'s
loop if you hit 429 errors.

## Multilingual support and summarization

**Cross-lingual Q&A**: the app uses a multilingual embedding model
(`paraphrase-multilingual-MiniLM-L12-v2`), which places semantically
similar text from *different* languages close together in the same vector
space. That means a question asked in French can retrieve the right chunks
from a Hindi video's transcript — no translation step is needed at
retrieval time. `rag/language.py` then detects the question's language
locally (via `langdetect`, free, no API call) and instructs the LLM to
answer in that language, translating the retrieved (possibly
different-language) context as part of generating the answer. If detection
isn't confident (e.g. very short questions), it falls back to an implicit
"answer in the same language as the question" instruction, which Llama
models handle reasonably well on their own.

⚠️ **If you're upgrading from an earlier version** that used the
English-only `all-MiniLM-L6-v2` model: the new multilingual model's
vectors aren't compatible with your old FAISS indexes. Run
`python -m scripts.reindex_all` once to rebuild everything from your
already-cached transcripts (fast — no re-downloading).

**Summarization**: retrieval-based Q&A is the wrong tool for "summarize
this video" — it only returns the top-k chunks most similar to the *word*
"summarize", not the whole transcript. `rag/summarize.py` instead does a
map-reduce pass: split the full transcript into a handful of larger
sections (size scales with video length, capped at roughly 12 LLM calls
even for very long videos), summarize each independently, then combine
those into one final summary. It's triggered two ways: a dedicated
"📋 Summarize this video" button, or just typing something like
"summarize this video" / "tl;dr" / "give me an overview" in chat
(`rag/summarize.is_summary_request()` — a keyword heuristic, not a full
intent classifier; it's tuned to avoid false positives on normal
questions, at the cost of occasionally missing an unusual phrasing).
Summarization only works for one selected video at a time — "summarize
all indexed videos" isn't a well-defined single summary, so the button is
disabled and chat requests are redirected when "All indexed videos" is
the active scope.

## Design notes

- **Speech-to-text fallback**: when a video has no YouTube captions,
  `rag/speech_to_text.py` downloads the audio (`yt-dlp`), splits it into
  ≤20-minute chunks to stay under Groq's free-tier 25MB per-request limit,
  and transcribes each chunk with Groq's free Whisper API
  (`whisper-large-v3-turbo`) — the same account/key already used for the
  LLM. Segment timestamps are offset per chunk and merged so the rest of
  the pipeline (chunking, timestamp-linked answers) works identically
  regardless of which transcript source was used. This only runs when
  captions are missing — captions are always tried first since they're
  free and instant.
- **Embeddings run locally** (`paraphrase-multilingual-MiniLM-L12-v2`) —
  no API cost or rate limits for the embedding step, only the final LLM
  call uses an external (free) API.
- **Grounding guard**: if the closest retrieved chunk's FAISS distance is
  above `MAX_DISTANCE_FOR_GROUNDED_ANSWER` (see `rag/config.py`), the app
  tells the user the video(s) don't cover the question instead of asking
  the LLM to answer anyway. This threshold was chosen empirically and
  should be re-tuned if you swap embedding models (use the eval harness's
  `out_of_scope_correct_refusal_rate` metric to check it).
- **Persistence**: transcripts and FAISS indexes are cached to disk, so
  re-indexing the same video, or restarting the app, doesn't require
  re-fetching or re-embedding.

## Limitations / known gaps

- Whisper fallback requires `ffmpeg` installed locally and a working
  internet connection to Groq's API; very long videos (multi-hour) will
  make several Whisper API calls, which is still within the free tier's
  daily limits for normal use but worth knowing about.
- English-only by default (`languages=["en", "en-US", "en-GB"]` in
  `transcript.py`) — pass other language codes to `fetch_transcript` for
  multilingual videos. Whisper itself is multilingual, so this is mostly a
  caption-language-preference setting, not a hard limit.
- FAISS distance thresholding is a heuristic, not a calibrated confidence
  score — the eval harness's `retrieval_localization_hit_rate` and
  `out_of_scope_correct_refusal_rate` metrics are how you validate/tune it.
- The LLM-as-judge scores are a cheap proxy for human judgment, not a
  substitute for it — spot-check a sample of judged answers by hand.

## Future improvements

- Hybrid search (BM25 + dense) with re-ranking
- Deployment to Streamlit Cloud / Hugging Face Spaces
- Translate system/refusal messages (currently English-only) into the
  detected question language too

## License

This project is for academic use only.
