"""Retrieval + grounded answer generation with source citations."""
import os
from dataclasses import dataclass
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS

from . import config
from .transcript import format_timestamp
from .language import target_language_instruction

PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant that answers questions using ONLY the
transcript excerpts provided below. Do not use outside knowledge and do not
guess. If the excerpts don't contain the answer, say clearly (in the
requested language) that the video(s) don't cover it.

The excerpts may be in a different language than the question — that's
fine, translate the relevant information as part of your answer rather than
quoting it in the original language.

Transcript excerpts:
{context}

Question: {question}

Answer {target_language}, staying strictly grounded in the excerpts above:"""
)


@dataclass
class Source:
    title: str
    url: str
    start: float
    timestamp: str
    snippet: str
    distance: float


@dataclass
class AnswerResult:
    answer: str
    sources: List[Source]
    grounded: bool
    context: str = ""


def get_llm():
    """Groq's free-tier API — OpenAI-compatible, no cost for normal usage.
    Get a free key at https://console.groq.com/keys and set GROQ_API_KEY."""
    from langchain_groq import ChatGroq

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Get a free key at "
            "https://console.groq.com/keys and set it as an environment "
            "variable (or paste it in the sidebar)."
        )
    return ChatGroq(model=config.GROQ_MODEL_NAME, temperature=0.2, api_key=api_key)


def _build_context(docs_with_scores) -> str:
    blocks = []
    for i, (doc, score) in enumerate(docs_with_scores, start=1):
        ts = format_timestamp(doc.metadata["start"])
        blocks.append(f"[{i}] ({doc.metadata['title']} @ {ts})\n{doc.page_content}")
    return "\n\n".join(blocks)


def answer_question(
    store: FAISS,
    question: str,
    k: int = config.TOP_K,
    max_distance: float = config.MAX_DISTANCE_FOR_GROUNDED_ANSWER,
) -> AnswerResult:
    docs_with_scores = store.similarity_search_with_score(question, k=k)

    if not docs_with_scores:
        return AnswerResult(
            answer="No indexed content to search yet — add a video first.",
            sources=[],
            grounded=False,
            context="",
        )

    best_distance = min(score for _, score in docs_with_scores)
    if best_distance > max_distance:
        return AnswerResult(
            answer=(
                "I couldn't find anything in the indexed video(s) that "
                "addresses this question. Try rephrasing, or index a video "
                "that covers this topic."
            ),
            sources=[],
            grounded=False,
            context=_build_context(docs_with_scores),
        )

    context = _build_context(docs_with_scores)
    chain = PROMPT | get_llm() | StrOutputParser()
    answer_text = chain.invoke({
        "context": context,
        "question": question,
        "target_language": target_language_instruction(question),
    })

    sources = [
        Source(
            title=doc.metadata["title"],
            url=f"{doc.metadata['url']}&t={int(doc.metadata['start'])}s"
            if "watch?v=" in doc.metadata["url"]
            else doc.metadata["url"],
            start=doc.metadata["start"],
            timestamp=format_timestamp(doc.metadata["start"]),
            snippet=doc.page_content[:220].strip() + "…",
            distance=round(float(score), 4),
        )
        for doc, score in docs_with_scores
    ]

    return AnswerResult(answer=answer_text, sources=sources, grounded=True, context=context)
