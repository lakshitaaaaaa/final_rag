# YouTube RAG Assistant

A production-oriented **Retrieval-Augmented Generation (RAG)** application that answers questions grounded in one or more YouTube videos. The system retrieves the most relevant transcript segments, generates fact-based responses with **timestamp citations**, and automatically falls back to speech-to-text transcription when captions are unavailable.

Unlike basic RAG demos, this project emphasizes **software engineering, retrieval quality, evaluation, multilingual support, and robustness**, making it suitable as an end-to-end AI application.

---

## Features

### AI-Powered Question Answering

* Ask natural language questions about any indexed YouTube video.
* Answers are generated **only from retrieved transcript context**.
* Every answer includes **timestamped citations** linking directly to the relevant part of the video.

### Multi-Video Knowledge Base

* Index multiple YouTube videos.
* Search either:

  * a specific video
  * the entire indexed library.

### Automatic Transcript Handling

* Fetches official YouTube transcripts when available.
* Caches transcripts locally to avoid repeated downloads.

### Whisper Fallback

If captions are unavailable:

* downloads video audio using `yt-dlp`
* splits long audio using `ffmpeg`
* transcribes using **Groq Whisper API**
* reconstructs timestamp-preserving transcript segments

This enables retrieval even for videos without subtitles.

### Persistent Vector Database

* FAISS vector indexes are stored on disk.
* Previously indexed videos do not require re-embedding.
* Supports both:

  * per-video indexes
  * combined library index.

### Grounded Generation

The application performs a similarity-distance check before sending retrieved context to the LLM.

If no sufficiently relevant information is found, the assistant refuses to answer instead of hallucinating.

### Multilingual Support

* Multilingual sentence embeddings
* Cross-lingual retrieval
* Ask questions in one language about videos in another language
* Responses are generated in the user's language.

### Whole Video Summarization

Supports complete video summarization using a **map-reduce pipeline**, avoiding the limitations of retrieval-only summarization.

### Evaluation Framework

Includes an evaluation harness for measuring:

* Correctness
* Faithfulness
* Retrieval localization accuracy
* Out-of-scope refusal rate

This enables systematic comparison of retrieval parameters and model configurations.

---

# Architecture

```text
                                                   YouTube URL
                               │
                               ▼
               Transcript Fetcher (cached)
                               │
               ┌───────────────┴───────────────┐
               │                               │
        Captions Available              No Captions
               │                               │
               ▼                               ▼
       Transcript Segments           yt-dlp Audio Download
               │                               │
               │                               ▼
               │                      ffmpeg Audio Split
               │                               │
               │                               ▼
               │                    Groq Whisper Transcription
               │                               │
               └───────────────┬───────────────┘
                               │
                               ▼
              Timestamp-Preserving Transcript Segments
                               │
                               ▼
                   Intelligent Text Chunking
                               │
                               ▼
              Multilingual Sentence Embeddings
                               │
                               ▼
                Persistent FAISS Vector Store
                               │
                               ▼
                 Similarity-Based Retrieval
                               │
                               ▼
                  Grounding Distance Guard
                               │
                               ▼
          Prompt Construction with Citations
                               │
                               ▼
               Groq Llama 3.1 8B Instant
                               │
                               ▼
        Grounded Answer + Timestamp References
```

---

# Tech Stack

## Languages

* Python

## Frameworks

* Streamlit
* LangChain

## LLM

* Groq Llama 3.1 8B Instant

## Embeddings

* Sentence Transformers
* paraphrase-multilingual-MiniLM-L12-v2

## Vector Database

* FAISS

## Speech-to-Text

* Groq Whisper API
* yt-dlp
* ffmpeg

## Libraries

* youtube-transcript-api
* langdetect
* HuggingFace Transformers
* requests

---

# Project Structure

```text
youtube-rag-assistant/

├── app.py
├── rag/
│   ├── config.py
│   ├── transcript.py
│   ├── speech_to_text.py
│   ├── language.py
│   ├── chunking.py
│   ├── vectorstore.py
│   ├── qa_chain.py
│   ├── summarize.py
│   └── ingest.py
│
├── scripts/
│   └── reindex_all.py
│
├── eval/
│   ├── metrics.py
│   ├── run_eval.py
│   ├── compare.py
│   └── eval_set.example.json
│
├── data/
│   ├── transcripts/
│   ├── audio/
│   ├── faiss_index/
│   └── manifest.json
│
├── requirements.txt
└── README.md
```

---

# Installation

## Clone the repository

```bash
git clone https://github.com/<your-username>/youtube-rag-assistant.git

cd youtube-rag-assistant
```

---

## Install dependencies

```bash
pip install -r requirements.txt
```

---

## Install FFmpeg

FFmpeg is required only when automatic Whisper transcription is needed.

### macOS

```bash
brew install ffmpeg
```

### Ubuntu

```bash
sudo apt install ffmpeg
```

### Windows

Install from:

https://ffmpeg.org/download.html

---

## Configure Environment Variables

Create a `.env` file.

```env
GROQ_API_KEY=your_api_key_here
```

---

## Run the application

```bash
streamlit run app.py
```

---

# Usage

1. Paste a YouTube URL.
2. Click **Index Video**.
3. Wait for transcript processing.
4. Ask questions in the chat interface.
5. Click **Summarize Video** to generate a complete summary.
6. Add additional videos to build a searchable knowledge base.

---

# Evaluation

The project includes an evaluation framework for measuring retrieval quality.

Metrics include:

* Correctness
* Faithfulness
* Retrieval localization
* Out-of-scope refusal rate

Run evaluation:

```bash
python -m eval.run_eval
```

Compare different retrieval configurations:

```bash
python -m eval.compare
```

This allows experimentation with:

* chunk size
* overlap
* retrieval depth (k)
* similarity thresholds
* embedding models

---

# Design Highlights

* Modular architecture
* Persistent caching
* Timestamp-preserving retrieval
* Automatic speech-to-text fallback
* Multilingual semantic search
* Grounded generation
* Hallucination mitigation
* Configurable evaluation pipeline
* Production-style project organization

---

# Future Improvements

### Deployment

* Deploy the application on **Streamlit Cloud**, **Hugging Face Spaces**, or **Render** for public access and recruiter demonstrations.

### User Management

* Add user authentication.
* Store user-specific indexed videos and chat history using a database.

### Containerization

* Package the application with **Docker** for consistent deployment across environments.

### CI/CD

* Implement automated testing and deployment using **GitHub Actions**.

### Observability

* Add structured logging.
* Integrate metrics collection.
* Include distributed tracing for easier debugging and monitoring.

### Streaming & Async Processing

* Support streaming LLM responses.
* Introduce asynchronous indexing and background processing for improved responsiveness.

### Hybrid Retrieval

* Combine **BM25 lexical search** with **dense vector retrieval**, followed by reranking for improved retrieval accuracy.

### Advanced Retrieval

* Context-aware reranking
* Metadata filtering
* Query rewriting
* Adaptive chunking
* Incremental index updates

### Scalability

* Replace local FAISS with a production vector database such as Pinecone, Weaviate, Milvus, or Qdrant.
* Add distributed indexing for large video collections.

---

# License

This project is intended for educational and academic purposes.
