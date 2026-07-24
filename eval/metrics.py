"""
LLM-as-judge metrics for the RAG pipeline.

Two things are scored, and they are deliberately kept separate because a
RAG system can fail in two different ways:

  - Faithfulness: is the generated answer actually supported by the
    retrieved context? (catches hallucination even when retrieval worked)
  - Correctness: does the answer match what a human would expect, given a
    reference answer? (catches retrieval failures / wrong chunks)

Both are scored 1-5 by the LLM itself (Groq/Llama), which is a common,
cheap proxy for human judgment. It's not perfect, but it's far better than
no evaluation at all, and it's standard practice (e.g. RAGAS uses the same
idea).
"""
import json
import re
from dataclasses import dataclass
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from rag.qa_chain import get_llm

FAITHFULNESS_PROMPT = ChatPromptTemplate.from_template(
    """You are grading whether an AI-generated answer is faithful to the
provided source context (i.e. not hallucinated).

Context the AI was given:
{context}

AI's answer:
{answer}

Score from 1 to 5:
5 = every claim in the answer is directly supported by the context
3 = mostly supported, but includes minor unsupported details
1 = the answer includes significant claims not found in the context at all

Respond with ONLY a JSON object: {{"score": <int 1-5>, "reason": "<one short sentence>"}}"""
)

CORRECTNESS_PROMPT = ChatPromptTemplate.from_template(
    """You are grading whether an AI-generated answer correctly addresses a
question, compared to a reference answer written by a human.

Question: {question}

Reference answer (human, ground truth): {expected_answer}

AI's answer: {answer}

Score from 1 to 5:
5 = fully correct, matches the substance of the reference answer
3 = partially correct, missing or slightly off on some details
1 = incorrect or does not address the question

Respond with ONLY a JSON object: {{"score": <int 1-5>, "reason": "<one short sentence>"}}"""
)


@dataclass
class JudgeScore:
    score: Optional[int]
    reason: str


def _parse_judge_json(raw: str) -> JudgeScore:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return JudgeScore(score=None, reason=f"Could not parse judge output: {raw[:200]}")
    try:
        data = json.loads(match.group(0))
        return JudgeScore(score=int(data.get("score")), reason=str(data.get("reason", "")))
    except (ValueError, TypeError, json.JSONDecodeError):
        return JudgeScore(score=None, reason=f"Could not parse judge output: {raw[:200]}")


def score_faithfulness(context: str, answer: str) -> JudgeScore:
    chain = FAITHFULNESS_PROMPT | get_llm() | StrOutputParser()
    raw = chain.invoke({"context": context, "answer": answer})
    return _parse_judge_json(raw)


def score_correctness(question: str, expected_answer: str, answer: str) -> JudgeScore:
    chain = CORRECTNESS_PROMPT | get_llm() | StrOutputParser()
    raw = chain.invoke(
        {"question": question, "expected_answer": expected_answer, "answer": answer}
    )
    return _parse_judge_json(raw)


def retrieval_localization_hit(
    sources, expected_timestamp_seconds: Optional[float], tolerance_seconds: float = 90.0
) -> Optional[bool]:
    """True if any retrieved chunk starts within `tolerance_seconds` of the
    expected timestamp. Returns None if no expected timestamp was given
    (i.e. this check doesn't apply to that question)."""
    if expected_timestamp_seconds is None:
        return None
    return any(
        abs(s.start - expected_timestamp_seconds) <= tolerance_seconds for s in sources
    )
