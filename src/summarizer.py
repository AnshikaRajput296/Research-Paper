"""
Summarizer using llama-3.1-8b-instant.

Fixes the '12834 > 1024' tokenizer error by completely removing
load_summarize_chain (which pipes all text through a HuggingFace tokenizer).
Instead uses direct LLM calls with hard character-based truncation at every
stage so nothing oversized ever reaches any tokenizer.

Also fixes 429 rate limit errors by:
- Processing chunks sequentially with a fixed delay between calls
- Reading Groq's suggested retry wait time from error messages
- Capping input/output tokens tightly per call
"""

import os
import re
import time
import random
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from src.llm import _get_api_key, _parse_suggested_wait

MODEL_NAME = "llama-3.1-8b-instant"

# ── Token budget (Groq free tier: 6000 TPM) ─────────────────────────────────
# ~450 input + 150 output = ~600 tokens per map call
# 7s delay between calls = ~8 calls/min = ~4800 tokens/min (safe under 6000)
MAP_INPUT_MAX_CHARS  = 1800   # ~450 tokens
MAP_OUTPUT_MAX_TOKENS = 150
MAP_CALL_DELAY        = 7.0   # seconds between sequential map calls

COMBINE_INPUT_MAX_CHARS   = 10000  # ~2500 tokens
COMBINE_OUTPUT_MAX_TOKENS = 800

MAX_RETRIES = 5
BASE_WAIT   = 3


MAP_PROMPT = """Extract key information from this research paper section. Short bullet points only. Include methods, numbers, results if present.

Section:
{text}

Key points:"""


COMBINE_PROMPT = """You are an expert research assistant. Using the notes below from every section of a research paper, write a COMPREHENSIVE STRUCTURED summary.

## Overview
3-4 sentences: what the paper is about, domain, significance.

## Problem Statement
The specific problem or gap addressed.

## Methodology
The approach, architecture, or method. Include datasets/tools if mentioned.

## Key Results & Findings
Bullet points with quantitative results and metrics.

## Contributions
3-5 distinct contributions.

## Limitations & Future Work
Stated limitations and future directions.

## Conclusion
2-3 sentence wrap-up.

Notes:
{notes}

Comprehensive structured summary:"""


def _llm(streaming: bool = False, max_tokens: int = 600) -> ChatGroq:
    return ChatGroq(
        model=MODEL_NAME,
        temperature=0.2,
        max_tokens=max_tokens,
        api_key=_get_api_key(),
        streaming=streaming,
        request_timeout=60,
    )


def _truncate(text: str, max_chars: int) -> str:
    """Hard-truncate to char budget at a word boundary."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_space = cut.rfind(" ")
    if last_space > 0:
        cut = cut[:last_space]
    return cut + " […truncated]"


def _invoke_with_retry(llm: ChatGroq, prompt: str) -> str:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return llm.invoke(prompt).content
        except Exception as e:
            err = str(e)
            is_rate_limit = "429" in err or "rate_limit" in err.lower()
            if is_rate_limit and attempt < MAX_RETRIES:
                wait = _parse_suggested_wait(err) or (BASE_WAIT * (2 ** (attempt - 1)) + random.uniform(0, 1))
                time.sleep(wait)
            else:
                raise


def _get_paper_docs(documents: list, paper_name: str) -> list[Document]:
    return [d for d in documents if d.metadata.get("paper", "") == paper_name]


def _sample_chunks(docs: list[Document], max_chunks: int = 15) -> list[Document]:
    """Evenly sample up to max_chunks across the full paper."""
    if len(docs) <= max_chunks:
        return docs
    step = len(docs) / max_chunks
    return [docs[int(i * step)] for i in range(max_chunks)]


def _map_stage(chunks: list[Document], progress_callback=None) -> str:
    """
    Summarise each chunk one at a time with a delay between calls.
    Sequential (not parallel) to stay under Groq's 6000 TPM free-tier limit.
    Does NOT use load_summarize_chain — avoids the HuggingFace tokenizer
    that causes the '12834 > 1024' error.
    """
    notes = []
    total = len(chunks)

    for i, chunk in enumerate(chunks):
        safe_text = _truncate(chunk.page_content.strip(), MAP_INPUT_MAX_CHARS)
        prompt = MAP_PROMPT.format(text=safe_text)
        llm = _llm(max_tokens=MAP_OUTPUT_MAX_TOKENS)

        try:
            note = _invoke_with_retry(llm, prompt)
        except Exception as e:
            note = f"[chunk {i+1} failed: {e}]"

        notes.append(f"[Section {i+1}/{total}]\n{note}")

        if progress_callback:
            progress_callback(i + 1, total)

        if i < total - 1:
            time.sleep(MAP_CALL_DELAY)

    return "\n\n".join(notes)


def summarize_single_paper_streaming(
    documents: list,
    paper_name: str,
    max_chunks: int = 15,
    progress_callback=None,
):
    """
    Generator — yields summary tokens as they stream in.
    Use with: st.write_stream(summarize_single_paper_streaming(...))
    """
    paper_docs = _get_paper_docs(documents, paper_name)

    if not paper_docs:
        yield f"No content found for: {paper_name}"
        return

    sampled = _sample_chunks(paper_docs, max_chunks)

    # MAP: direct LLM calls, no HuggingFace tokenizer involved
    notes_text = _map_stage(sampled, progress_callback=progress_callback)

    # Hard cap before combine — prevents oversized combine input
    notes_text = _truncate(notes_text, COMBINE_INPUT_MAX_CHARS)

    # Brief pause so combine call doesn't land in the same TPM window
    time.sleep(3)

    # COMBINE: streamed
    llm = _llm(streaming=True, max_tokens=COMBINE_OUTPUT_MAX_TOKENS)
    prompt = COMBINE_PROMPT.format(notes=notes_text)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            for chunk in llm.stream(prompt):
                if chunk.content:
                    yield chunk.content
            return
        except Exception as e:
            err = str(e)
            is_rate_limit = "429" in err or "rate_limit" in err.lower()
            if is_rate_limit and attempt < MAX_RETRIES:
                wait = _parse_suggested_wait(err) or (BASE_WAIT * (2 ** (attempt - 1)))
                yield f"\n\n_(Rate limit hit — retrying in {wait:.0f}s…)_\n\n"
                time.sleep(wait)
            else:
                yield f"\n\n**Summary failed:** {e}"
                return


def summarize_single_paper(documents: list, paper_name: str, max_chunks: int = 15) -> str:
    """Non-streaming version. Collects the full generator output."""
    return "".join(summarize_single_paper_streaming(documents, paper_name, max_chunks))


def summarize_all_papers(documents: list) -> dict[str, str]:
    paper_names = sorted({d.metadata.get("paper", "Unknown Paper") for d in documents})
    summaries = {}
    for i, name in enumerate(paper_names):
        if i > 0:
            time.sleep(15)  # let TPM window reset between full papers
        summaries[name] = summarize_single_paper(documents, name)
    return summaries


def compare_papers(documents: list) -> str:
    paper_names = sorted({d.metadata.get("paper", "Unknown Paper") for d in documents})

    if len(paper_names) < 2:
        return "At least 2 papers are needed for comparison."

    llm = _llm(max_tokens=900)

    paper_contexts = []
    for name in paper_names:
        chunks = _get_paper_docs(documents, name)[:3]
        combined = "\n\n".join(_truncate(c.page_content, 1200) for c in chunks)
        paper_contexts.append(f"### {name}\n{combined}")

    all_context = _truncate("\n\n---\n\n".join(paper_contexts), COMBINE_INPUT_MAX_CHARS)

    prompt = f"""Write a structured comparative analysis:

## Research Goals
## Methodology Comparison
## Results Comparison
## Strengths & Weaknesses
## Key Similarities & Differences
## Overall Verdict

Papers:
{all_context}

Comparative analysis:"""

    return _invoke_with_retry(llm, prompt)