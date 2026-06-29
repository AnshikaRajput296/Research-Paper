import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document


def _get_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        max_tokens=512,
        api_key=os.environ.get("GROQ_API_KEY"),
    )


# ---------------------------------------------------------------------------
# Map prompt — summarise each chunk individually
# ---------------------------------------------------------------------------
MAP_PROMPT_TEMPLATE = """You are summarising a section of a research paper.
Extract ALL key information from this chunk. Be thorough — do not skip details.

Chunk:
{text}

Detailed notes from this chunk:"""

MAP_PROMPT = PromptTemplate(
    input_variables=["text"],
    template=MAP_PROMPT_TEMPLATE,
)

# ---------------------------------------------------------------------------
# Combine prompt — stitch all chunk summaries into one structured summary
# ---------------------------------------------------------------------------
COMBINE_PROMPT_TEMPLATE = """You are an expert research assistant.
Below are detailed notes extracted from every section of a research paper.
Write a COMPREHENSIVE, STRUCTURED summary of the ENTIRE paper using ALL notes.

Your summary MUST include these sections (use markdown headers):

## Overview
A clear 3-5 sentence description of what the paper is about, its domain, and its significance.

## Problem Statement
What specific problem or gap in knowledge does the paper address?

## Proposed Solution / Methodology
Describe the approach, architecture, algorithm, or method proposed in detail.
Include datasets, tools, frameworks, or experimental setup if mentioned.

## Key Results & Findings
List quantitative results, performance metrics, comparisons, and major findings.
Use bullet points and include numbers wherever available.

## Contributions
What does this paper contribute to the field? List 3-5 distinct contributions.

## Limitations & Future Work
What are the stated limitations? What future directions are suggested?

## Conclusion
A 2-3 sentence wrap-up of the paper's impact and take-away message.

---

Notes from all sections:
{text}

Comprehensive structured summary:"""

COMBINE_PROMPT = PromptTemplate(
    input_variables=["text"],
    template=COMBINE_PROMPT_TEMPLATE,
)


def _get_paper_docs(documents: list, paper_name: str) -> list[Document]:
    """Filter and return all document chunks belonging to a specific paper."""
    return [
        doc
        for doc in documents
        if doc.metadata.get("paper", "") == paper_name
    ]


def summarize_single_paper(documents: list, paper_name: str) -> str:
    """
    Generate a comprehensive structured summary of a single paper
    using a map-reduce chain that covers ALL chunks.
    """
    paper_docs = _get_paper_docs(documents, paper_name)

    if not paper_docs:
        return f"No content found for paper: {paper_name}"

    llm = _get_llm()

    chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt=MAP_PROMPT,
        combine_prompt=COMBINE_PROMPT,
        verbose=False,
    )

    result = chain.invoke({"input_documents": paper_docs})
    return result["output_text"]


def summarize_all_papers(documents: list) -> dict[str, str]:
    """
    Generate comprehensive summaries for every paper in the document set.
    Returns a dict: {paper_name: summary_text}
    """
    paper_names = sorted(
        {
            doc.metadata.get("paper", "Unknown Paper")
            for doc in documents
        }
    )

    summaries = {}
    for paper_name in paper_names:
        summaries[paper_name] = summarize_single_paper(documents, paper_name)

    return summaries


def compare_papers(documents: list) -> str:
    """
    Generate a structured comparative analysis across all uploaded papers.
    """
    paper_names = sorted(
        {
            doc.metadata.get("paper", "Unknown Paper")
            for doc in documents
        }
    )

    if len(paper_names) < 2:
        return "At least 2 papers are needed for comparison."

    llm = _get_llm()

    # Build a rich context: first 6 chunks from each paper
    paper_contexts = []
    for name in paper_names:
        chunks = _get_paper_docs(documents, name)[:6]
        combined_text = "\n\n".join(c.page_content for c in chunks)
        paper_contexts.append(f"### {name}\n{combined_text}")

    all_context = "\n\n---\n\n".join(paper_contexts)

    compare_prompt = f"""You are an expert research analyst.
Below are excerpts from multiple research papers. 
Write a DETAILED COMPARATIVE ANALYSIS covering:

## Research Goals
How does each paper's objective differ or align?

## Methodology Comparison
Compare approaches, algorithms, datasets, and experimental setups.

## Results Comparison
Compare reported metrics, benchmarks, and performance.

## Strengths & Weaknesses
For each paper, note key strengths and limitations.

## Key Similarities
What common themes, techniques, or conclusions emerge?

## Key Differences
What fundamentally distinguishes these papers?

## Overall Verdict
Which paper(s) contribute most significantly and why?

Papers to compare:
{all_context}

Detailed comparative analysis:"""

    response = llm.invoke(compare_prompt)
    return response.content
