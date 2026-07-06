"""
RAGAS Evaluation Module
=======================
Evaluates the RAG pipeline using 4 core RAGAS metrics:
  - Answer Relevancy
  - Faithfulness
  - Context Precision
  - Context Recall

Usage (from Streamlit sidebar or standalone):
    from src.ragas_eval import run_ragas_evaluation
    results = run_ragas_evaluation(qa_chain, documents)
"""

import os
import json
from datetime import datetime
from langchain_groq import ChatGroq


# ---------------------------------------------------------------------------
# Default test questions — override by passing your own list
# ---------------------------------------------------------------------------
DEFAULT_TEST_QUESTIONS = [
    "What is the main objective of this research paper?",
    "What methodology or approach does the paper propose?",
    "What datasets were used in the experiments?",
    "What are the key results and performance metrics reported?",
    "What are the limitations mentioned in the paper?",
    "What future work do the authors suggest?",
    "What problem does this paper solve?",
    "How does the proposed method compare to existing baselines?",
    "What are the main contributions of this paper?",
    "What conclusions do the authors draw?",
]


def _safe_score(value, default=0.0) -> float:
    try:
        return round(float(value), 4)
    except Exception:
        return default


def _evaluate_single(
    llm: ChatGroq,
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str = "",
) -> dict:
    """
    Use LLM-as-judge to score one QA sample on 4 RAGAS-style metrics.
    Returns scores between 0 and 1 for each metric.
    """

    context_block = "\n\n---\n\n".join(contexts[:4])  # top 4 chunks

    prompt = f"""You are an expert evaluator for RAG (Retrieval-Augmented Generation) systems.
Evaluate the following QA sample on exactly 4 metrics. 
Respond ONLY with a valid JSON object — no prose, no markdown, no explanation.

Question: {question}

Retrieved Contexts:
{context_block}

Generated Answer: {answer}

Score each metric from 0.0 to 1.0:

1. answer_relevancy: Does the answer directly address the question? (1.0 = perfectly relevant)
2. faithfulness: Is every claim in the answer supported by the provided contexts? (1.0 = fully grounded)
3. context_precision: Are the retrieved contexts precise and mostly relevant to the question? (1.0 = all contexts relevant)
4. context_recall: Do the contexts contain all information needed to answer the question? (1.0 = complete coverage)

Respond with ONLY this JSON (no other text):
{{
  "answer_relevancy": <float 0-1>,
  "faithfulness": <float 0-1>,
  "context_precision": <float 0-1>,
  "context_recall": <float 0-1>,
  "reasoning": "<one sentence explaining your scores>"
}}"""

    try:
        response = llm.invoke(prompt)
        raw = response.content.strip()

        # Strip any accidental markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        parsed = json.loads(raw.strip())
        return {
            "answer_relevancy": _safe_score(parsed.get("answer_relevancy")),
            "faithfulness": _safe_score(parsed.get("faithfulness")),
            "context_precision": _safe_score(parsed.get("context_precision")),
            "context_recall": _safe_score(parsed.get("context_recall")),
            "reasoning": parsed.get("reasoning", ""),
        }
    except Exception as e:
        return {
            "answer_relevancy": 0.0,
            "faithfulness": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "reasoning": f"Evaluation failed: {e}",
        }


def run_ragas_evaluation(
    qa_chain,
    questions: list[str] | None = None,
    progress_callback=None,
) -> dict:
    """
    Run RAGAS-style evaluation across a set of test questions.

    Args:
        qa_chain:          The built LangChain retrieval chain.
        questions:         List of test questions (uses defaults if None).
        progress_callback: Optional callable(current, total, question) for UI updates.

    Returns:
        {
            "metrics": {               # average scores
                "answer_relevancy": float,
                "faithfulness": float,
                "context_precision": float,
                "context_recall": float,
            },
            "per_question": [          # per-sample breakdown
                {
                    "question": str,
                    "answer": str,
                    "scores": {...},
                    "reasoning": str,
                }
            ],
            "num_questions": int,
            "timestamp": str,
        }
    """
    if questions is None:
        questions = DEFAULT_TEST_QUESTIONS

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.0,
        api_key=os.environ.get("GROQ_API_KEY"),
    )

    all_scores = {
        "answer_relevancy": [],
        "faithfulness": [],
        "context_precision": [],
        "context_recall": [],
    }
    per_question_results = []

    for i, question in enumerate(questions):
        if progress_callback:
            progress_callback(i + 1, len(questions), question)

        try:
            # Run the RAG chain
            response = qa_chain(
                {"input": question, "chat_history": ""}
            )
            answer = response.get("answer", "")
            context_docs = response.get("context", [])
            contexts = [doc.page_content for doc in context_docs]

            # Score with LLM-as-judge
            scores = _evaluate_single(llm, question, answer, contexts)

            for metric in all_scores:
                all_scores[metric].append(scores[metric])

            per_question_results.append(
                {
                    "question": question,
                    "answer": answer[:400] + ("..." if len(answer) > 400 else ""),
                    "scores": {k: scores[k] for k in all_scores},
                    "reasoning": scores.get("reasoning", ""),
                }
            )

        except Exception as e:
            per_question_results.append(
                {
                    "question": question,
                    "answer": "ERROR",
                    "scores": {k: 0.0 for k in all_scores},
                    "reasoning": str(e),
                }
            )
            for metric in all_scores:
                all_scores[metric].append(0.0)

    # Compute averages
    n = len(questions)
    avg_metrics = {
        metric: round(sum(scores) / n, 4) if n > 0 else 0.0
        for metric, scores in all_scores.items()
    }

    return {
        "metrics": avg_metrics,
        "per_question": per_question_results,
        "num_questions": n,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
