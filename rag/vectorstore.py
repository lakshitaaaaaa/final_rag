"""
Persistent FAISS vector store.

Every indexed video is stored twice on disk:
  1. data/faiss_index/<video_id>/   -> lets you search a single video only
  2. data/faiss_index/_all/         -> a combined index across every video
     that's ever been indexed, for cross-video / channel-style questions

A manifest.json tracks which videos have been indexed so the UI can list them.
"""
import json
from functools import lru_cache
from typing import List, Dict, Optional

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from . import config


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Loaded once per process — this model runs locally, no API calls, no cost."""
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)


def _load_manifest() -> Dict:
    if config.MANIFEST_PATH.exists():
        return json.loads(config.MANIFEST_PATH.read_text(encoding="utf-8"))
    return {}


def _save_manifest(manifest: Dict) -> None:
    config.MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def list_indexed_videos() -> Dict:
    return _load_manifest()


def _make_documents(video_id: str, title: str, url: str, chunks: List[Dict]) -> List[Document]:
    docs = []
    for idx, c in enumerate(chunks):
        docs.append(
            Document(
                page_content=c["text"],
                metadata={
                    "video_id": video_id,
                    "title": title,
                    "url": url,
                    "start": c["start"],
                    "chunk_index": idx,
                },
            )
        )
    return docs


def index_video(
    video_id: str,
    title: str,
    url: str,
    chunks: List[Dict],
    index_root=None,
    update_manifest: bool = True,
    transcript_source: str = "youtube_captions",
) -> int:
    """Embeds chunks and (re)builds the per-video index, then merges into the
    combined index. Returns the number of chunks indexed.

    index_root: override the default data/faiss_index directory. Used by the
    eval harness to build isolated indexes (e.g. per chunk-size experiment)
    without touching the app's live indexes. update_manifest is disabled for
    those isolated runs too, so experiments never show up in the app's UI.
    """
    index_root = index_root or config.FAISS_INDEX_DIR
    embeddings = get_embeddings()
    docs = _make_documents(video_id, title, url, chunks)
    if not docs:
        return 0

    # Per-video index (overwritten if this video is re-indexed)
    per_video_store = FAISS.from_documents(docs, embeddings)
    per_video_dir = index_root / video_id
    per_video_store.save_local(str(per_video_dir))

    # Combined index across all videos under this root
    all_dir = index_root / "_all"
    if all_dir.exists():
        all_store = FAISS.load_local(
            str(all_dir), embeddings, allow_dangerous_deserialization=True
        )
        all_store.merge_from(per_video_store)
    else:
        all_store = per_video_store
    all_store.save_local(str(all_dir))

    if update_manifest:
        manifest = _load_manifest()
        manifest[video_id] = {
            "title": title, "url": url, "num_chunks": len(docs),
            "transcript_source": transcript_source,
        }
        _save_manifest(manifest)

    return len(docs)


def load_store(video_id: Optional[str] = None, index_root=None) -> Optional[FAISS]:
    """video_id=None loads the combined cross-video index. index_root overrides
    the default data/faiss_index directory (see index_video)."""
    index_root = index_root or config.FAISS_INDEX_DIR
    embeddings = get_embeddings()
    target_dir = index_root / (video_id if video_id else "_all")
    if not target_dir.exists():
        return None
    return FAISS.load_local(str(target_dir), embeddings, allow_dangerous_deserialization=True)
