"""End-to-end: URL -> cached transcript -> timestamped chunks -> FAISS index."""
from dataclasses import dataclass

from . import transcript, chunking, vectorstore, config


@dataclass
class IngestResult:
    video_id: str
    title: str
    url: str
    num_chunks: int
    already_indexed: bool
    transcript_source: str = "youtube_captions"


def ingest_video(
    url_or_id: str,
    force: bool = False,
    chunk_size: int = config.CHUNK_SIZE_CHARS,
    chunk_overlap: int = config.CHUNK_OVERLAP_CHARS,
    index_root=None,
    update_manifest: bool = True,
) -> IngestResult:
    """chunk_size/chunk_overlap/index_root/update_manifest let the eval
    harness run the same video through different chunking configs into
    isolated, disposable indexes (see rag/vectorstore.py)."""
    video_id = transcript.extract_video_id(url_or_id)

    if update_manifest and not index_root:
        manifest = vectorstore.list_indexed_videos()
        if video_id in manifest and not force:
            m = manifest[video_id]
            return IngestResult(
                video_id=video_id,
                title=m["title"],
                url=m["url"],
                num_chunks=m["num_chunks"],
                already_indexed=True,
                transcript_source=m.get("transcript_source", "youtube_captions"),
            )

    url = f"https://www.youtube.com/watch?v={video_id}"
    title = transcript.fetch_video_title(video_id)
    segments = transcript.fetch_transcript(video_id)
    source = transcript.get_transcript_source(video_id) or "youtube_captions"
    chunks = chunking.chunk_transcript(segments, chunk_size, chunk_overlap)
    num_chunks = vectorstore.index_video(
        video_id, title, url, chunks,
        index_root=index_root, update_manifest=update_manifest,
        transcript_source=source,
    )

    return IngestResult(
        video_id=video_id, title=title, url=url, num_chunks=num_chunks,
        already_indexed=False, transcript_source=source,
    )
