import os
import streamlit as st
from dotenv import load_dotenv

from rag import ingest, vectorstore, qa_chain, summarize
from rag.language import target_language_instruction

load_dotenv()

st.set_page_config(page_title="YouTube RAG Assistant", page_icon="🎬", layout="wide")

# ---------------------------------------------------------------------------
# Sidebar: API key + video management
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Setup")

    default_key = os.environ.get("GROQ_API_KEY", "")
    groq_key_input = st.text_input(
        "Groq API key (free)", value=default_key, type="password",
        help="Get a free key at https://console.groq.com/keys",
    )
    if groq_key_input:
        os.environ["GROQ_API_KEY"] = groq_key_input

    st.divider()
    st.header("Add a video")
    st.caption("Works with videos in any language — captions are used when "
               "available, otherwise Whisper transcribes the audio.")
    new_url = st.text_input("YouTube URL or video ID", key="new_url")
    if st.button("Index video", use_container_width=True):
        if not new_url.strip():
            st.warning("Paste a YouTube URL first.")
        else:
            with st.spinner("Fetching transcript, chunking, and embedding…"):
                try:
                    result = ingest.ingest_video(new_url)
                    src_label = (
                        "🎙️ transcribed with Whisper (no captions available)"
                        if result.transcript_source == "whisper_fallback"
                        else "📝 YouTube captions"
                    )
                    if result.already_indexed:
                        st.info(f"'{result.title}' is already indexed ({result.num_chunks} chunks, {src_label}).")
                    else:
                        st.success(f"Indexed '{result.title}' — {result.num_chunks} chunks, {src_label}.")
                except Exception as e:
                    st.error(str(e))

    st.divider()
    st.header("Indexed videos")
    manifest = vectorstore.list_indexed_videos()
    if not manifest:
        st.caption("No videos indexed yet.")
    else:
        for vid, meta in manifest.items():
            src_icon = "🎙️" if meta.get("transcript_source") == "whisper_fallback" else "📝"
            st.caption(f"{src_icon} {meta['title']}  \n{meta['num_chunks']} chunks")

# ---------------------------------------------------------------------------
# Main: search scope + chat
# ---------------------------------------------------------------------------
st.title("🎬 YouTube RAG Assistant")
st.caption(
    "Ask questions grounded in the transcripts of the videos you've indexed — "
    "in any language, regardless of the video's own language."
)

manifest = vectorstore.list_indexed_videos()

if not manifest:
    st.info("Add a video from the sidebar to get started.")
    st.stop()

scope_options = {"All indexed videos": None}
scope_options.update({meta["title"]: vid for vid, meta in manifest.items()})
col1, col2 = st.columns([3, 1])
with col1:
    scope_label = st.selectbox("Search scope", list(scope_options.keys()))
scope_video_id = scope_options[scope_label]

# Summarize button — only meaningful for a single selected video, since
# "summarize all indexed videos" isn't a well-defined single summary.
with col2:
    st.write("")  # vertical alignment spacer
    summarize_clicked = st.button(
        "📋 Summarize this video",
        use_container_width=True,
        disabled=scope_video_id is None,
        help="Select a single video above (not 'All indexed videos') to enable this."
    )

if scope_video_id is None:
    st.caption("Select a single video above to enable one-click summarization.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for role, content in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(content)


def run_summary(video_id: str, target_language: str):
    with st.chat_message("assistant"):
        with st.spinner("Reading through the full transcript and summarizing…"):
            try:
                result = summarize.summarize_video(video_id, target_language=target_language)
                st.markdown(result["summary"])
                st.caption(f"Summarized from {result['num_sections']} section(s) of the transcript.")
                st.session_state.chat_history.append(("assistant", result["summary"]))
            except Exception as e:
                st.error(str(e))


if summarize_clicked and scope_video_id:
    title = manifest[scope_video_id]["title"]
    user_msg = f"Summarize '{title}'"
    st.session_state.chat_history.append(("user", user_msg))
    with st.chat_message("user"):
        st.markdown(user_msg)
    run_summary(scope_video_id, target_language="in English")

question = st.chat_input("Ask something, or say 'summarize this video'…")
if question:
    st.session_state.chat_history.append(("user", question))
    with st.chat_message("user"):
        st.markdown(question)

    if not os.environ.get("GROQ_API_KEY"):
        with st.chat_message("assistant"):
            st.error("Add a free Groq API key in the sidebar first.")
    elif summarize.is_summary_request(question):
        if scope_video_id is None:
            with st.chat_message("assistant"):
                msg = ("Summarizing works best for one video at a time — "
                       "pick a specific video from the 'Search scope' dropdown above, "
                       "then ask again.")
                st.warning(msg)
                st.session_state.chat_history.append(("assistant", msg))
        else:
            run_summary(scope_video_id, target_language=target_language_instruction(question))
    else:
        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant transcript chunks and generating an answer…"):
                try:
                    store = vectorstore.load_store(scope_video_id)
                    result = qa_chain.answer_question(store, question)
                    st.markdown(result.answer)
                    if result.sources:
                        with st.expander(f"Sources ({len(result.sources)})"):
                            for s in result.sources:
                                st.markdown(
                                    f"**{s.title}** @ {s.timestamp} "
                                    f"(distance {s.distance}) — [jump to moment]({s.url})"
                                )
                                st.caption(s.snippet)
                    st.session_state.chat_history.append(("assistant", result.answer))
                except Exception as e:
                    st.error(str(e))
