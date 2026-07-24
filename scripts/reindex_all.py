"""
Rebuilds every indexed video's FAISS index from its cached transcript.

Run this once after switching EMBEDDING_MODEL_NAME in rag/config.py (e.g.
after this project moved from the English-only MiniLM model to the
multilingual one) — old FAISS vectors aren't compatible with a new
embedding model, so they need to be re-embedded, not just re-loaded.

This does NOT re-fetch transcripts or re-download anything from YouTube —
it reuses what's already cached in data/transcripts/, so it's fast and free.

Usage:
    python -m scripts.reindex_all
"""
import shutil

from rag import config, vectorstore, ingest


def main():
    manifest = vectorstore.list_indexed_videos()
    if not manifest:
        print("No indexed videos found — nothing to reindex.")
        return

    print(f"Found {len(manifest)} indexed video(s). Rebuilding with model "
          f"'{config.EMBEDDING_MODEL_NAME}'...\n")

    if config.FAISS_INDEX_DIR.exists():
        shutil.rmtree(config.FAISS_INDEX_DIR)
        config.FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)

    for video_id, meta in manifest.items():
        print(f"Reindexing '{meta['title']}' ({video_id}) ...")
        ingest.ingest_video(meta["url"], force=True)

    print("\nDone. All videos rebuilt with the current embedding model.")


if __name__ == "__main__":
    main()
