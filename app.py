import streamlit as st

from src.ingest import load_documents, split_documents
from src.retriever import create_vectorstore
from src.guardrails import is_prompt_injection, validate_response_scope
from src.qa_chain import build_qa_chain
from src.summarizer import summarize_single_paper, summarize_all_papers, compare_papers
from src.utils import save_uploaded_files
from src.citations import format_citations
from src.ragas_eval import run_ragas_evaluation, DEFAULT_TEST_QUESTIONS

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Research Paper QA Assistant",
    layout="wide",
    page_icon="📄",
)

st.title(" Research Paper QA Assistant")

st.markdown(
    """
    Upload one or more research papers and:
    - **Ask questions** using Retrieval-Augmented Generation (RAG)
    - **Generate detailed summaries** covering the full paper
    - **Compare papers** side by side
    - **Evaluate RAG quality** using RAGAS metrics
    """
)

# ==================================================
# SESSION STATE
# ==================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = False
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "documents" not in st.session_state:
    st.session_state.documents = None
if "summary_cache" not in st.session_state:
    st.session_state.summary_cache = {}
if "comparison_cache" not in st.session_state:
    st.session_state.comparison_cache = None
if "all_summaries_cache" not in st.session_state:
    st.session_state.all_summaries_cache = None
if "ragas_results" not in st.session_state:
    st.session_state.ragas_results = None

# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.header(" Document Upload")

    uploaded_files = st.file_uploader(
        "Upload PDF Research Papers",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.button(" Process Papers", use_container_width=True):
            with st.spinner("Processing papers…"):
                pdf_paths = save_uploaded_files(uploaded_files)
                documents = load_documents(pdf_paths)
                chunks = split_documents(documents)
                create_vectorstore(chunks)

                st.session_state.qa_chain = build_qa_chain()
                st.session_state.documents = documents
                st.session_state.vectorstore_ready = True

                # Reset caches on new upload
                st.session_state.messages = []
                st.session_state.summary_cache = {}
                st.session_state.comparison_cache = None
                st.session_state.all_summaries_cache = None
                st.session_state.ragas_results = None

            st.success(
                f"✅ Processed {len(uploaded_files)} paper(s) — "
                f"{len(chunks)} chunks indexed."
            )

    st.divider()

    if st.session_state.vectorstore_ready:
        st.markdown("** Index Stats**")
        paper_names = sorted(
            {
                doc.metadata.get("paper", "Unknown")
                for doc in st.session_state.documents
            }
        )
        st.markdown(f"- Papers: **{len(paper_names)}**")
        st.markdown(f"- Chunks: **{len(st.session_state.documents)}**")
        for p in paper_names:
            st.markdown(f"  - {p}")

        st.divider()

    if st.button(" Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ==================================================
# MODE SELECTION
# ==================================================

mode = st.radio(
    "Select Mode",
    ["💬 Question Answering", "📝 Paper Summary", "📊 RAGAS Evaluation"],
    horizontal=True,
)

# ==================================================
# QUESTION ANSWERING MODE
# ==================================================

if mode == "💬 Question Answering":

    if not st.session_state.vectorstore_ready:
        st.info("Upload and process PDFs first using the sidebar.")
    else:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        user_question = st.chat_input("Ask a question about the papers…")

        if user_question:

            if is_prompt_injection(user_question):
                st.error(" Potential prompt injection detected. Please ask a genuine question.")
                st.stop()

            st.session_state.messages.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)

            history_text = "\n".join(
                [
                    f"{msg['role']}: {msg['content']}"
                    for msg in st.session_state.messages[-6:]
                ]
            )

            with st.chat_message("assistant"):
                with st.spinner("Searching papers…"):
                    response = st.session_state.qa_chain.invoke(
                        {"input": user_question, "chat_history": history_text}
                    )

                    answer = response["answer"]
                    context_docs = response.get("context", [])

                    if not context_docs:
                        answer = " No relevant information found in uploaded papers."
                    elif not validate_response_scope(answer):
                        answer = (
                            " The uploaded papers do not contain sufficient "
                            "information to answer this question."
                        )

                    st.markdown(answer)

                    citations = format_citations(context_docs)
                    if citations:
                        st.markdown("###  Sources")
                        for citation in citations:
                            st.markdown(f"- {citation}")

            st.session_state.messages.append({"role": "assistant", "content": answer})

# ==================================================
# SUMMARY MODE
# ==================================================

elif mode == "📝 Paper Summary":

    if not st.session_state.documents:
        st.info("⬅️ Upload and process PDFs first using the sidebar.")
    else:
        summary_mode = st.radio(
            "Summary Type",
            ["Single Paper Summary", "All Papers Summary", "Comparative Analysis"],
            horizontal=True,
        )

        paper_names = sorted(
            {
                doc.metadata.get("paper", "Unknown Paper")
                for doc in st.session_state.documents
            }
        )

        # ----------------------------------
        # SINGLE PAPER
        # ----------------------------------
        if summary_mode == "Single Paper Summary":
            selected_paper = st.selectbox("Select Paper", paper_names)

            if st.button(" Generate Summary"):
                if selected_paper not in st.session_state.summary_cache:
                    with st.spinner(
                        f"Reading all pages of '{selected_paper}' and generating summary…"
                    ):
                        summary = summarize_single_paper(
                            st.session_state.documents, selected_paper
                        )
                        st.session_state.summary_cache[selected_paper] = summary
                else:
                    st.info("Showing cached summary.")

                st.markdown(f"#  {selected_paper}")
                st.markdown(st.session_state.summary_cache[selected_paper])

        # ----------------------------------
        # ALL PAPERS
        # ----------------------------------
        elif summary_mode == "All Papers Summary":
            if st.button(" Generate All Summaries"):
                if st.session_state.all_summaries_cache is None:
                    with st.spinner("Generating comprehensive summaries for all papers…"):
                        st.session_state.all_summaries_cache = summarize_all_papers(
                            st.session_state.documents
                        )
                else:
                    st.info("Showing cached summaries.")

                for paper_name, summary in st.session_state.all_summaries_cache.items():
                    st.markdown(f"# 📄 {paper_name}")
                    st.markdown(summary)
                    st.divider()

        # ----------------------------------
        # COMPARATIVE
        # ----------------------------------
        else:
            if len(paper_names) < 2:
                st.warning("⚠️ Upload at least 2 papers to compare.")
            else:
                st.markdown(f"Comparing **{len(paper_names)} papers**: {', '.join(paper_names)}")
                if st.button(" Compare Papers"):
                    if st.session_state.comparison_cache is None:
                        with st.spinner("Performing comparative analysis…"):
                            st.session_state.comparison_cache = compare_papers(
                                st.session_state.documents
                            )
                    else:
                        st.info("Showing cached comparison.")

                    st.markdown("#  Comparative Analysis")
                    st.markdown(st.session_state.comparison_cache)

# ==================================================
# RAGAS EVALUATION MODE
# ==================================================

elif mode == "📊 RAGAS Evaluation":

    if not st.session_state.vectorstore_ready:
        st.info("⬅️ Upload and process PDFs first using the sidebar.")
    else:
        st.markdown("##  RAGAS Pipeline Evaluation")
        st.markdown(
            """
            Evaluate your RAG pipeline quality using **4 key metrics**:
            | Metric | What it measures |
            |---|---|
            | **Answer Relevancy** | Does the answer address the question? |
            | **Faithfulness** | Is the answer grounded in the retrieved context? |
            | **Context Precision** | Are retrieved chunks mostly relevant? |
            | **Context Recall** | Do the chunks cover all needed information? |
            """
        )

        st.divider()

        # Custom question editor
        with st.expander(" Customize Test Questions", expanded=False):
            st.markdown("Edit or add questions below (one per line):")
            default_text = "\n".join(DEFAULT_TEST_QUESTIONS)
            custom_questions_raw = st.text_area(
                "Test Questions",
                value=default_text,
                height=300,
                label_visibility="collapsed",
            )
            test_questions = [
                q.strip()
                for q in custom_questions_raw.strip().split("\n")
                if q.strip()
            ]
            st.markdown(f"**{len(test_questions)} questions** will be evaluated.")

        if st.button(" Run RAGAS Evaluation", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(current, total, question):
                progress_bar.progress(current / total)
                status_text.markdown(
                    f"**Evaluating {current}/{total}:** _{question[:80]}…_"
                )

            with st.spinner("Running evaluation…"):
                st.session_state.ragas_results = run_ragas_evaluation(
                    qa_chain=st.session_state.qa_chain,
                    questions=test_questions,
                    progress_callback=update_progress,
                )

            progress_bar.progress(1.0)
            status_text.markdown(" Evaluation complete!")

        # Display results
        if st.session_state.ragas_results:
            results = st.session_state.ragas_results
            metrics = results["metrics"]

            st.divider()
            st.markdown("###  Overall Scores")
            st.caption(f"Evaluated {results['num_questions']} questions · {results['timestamp']}")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric(
                "Answer Relevancy",
                f"{metrics['answer_relevancy']:.2f}",
                help="How well the answer addresses the question (0–1)",
            )
            col2.metric(
                "Faithfulness",
                f"{metrics['faithfulness']:.2f}",
                help="How grounded the answer is in retrieved context (0–1)",
            )
            col3.metric(
                "Context Precision",
                f"{metrics['context_precision']:.2f}",
                help="How relevant the retrieved chunks are (0–1)",
            )
            col4.metric(
                "Context Recall",
                f"{metrics['context_recall']:.2f}",
                help="How completely the chunks cover the answer (0–1)",
            )

            # Overall quality indicator
            avg_score = sum(metrics.values()) / len(metrics)
            quality = (
                "🟢 Excellent" if avg_score >= 0.80
                else "🟡 Good" if avg_score >= 0.65
                else "🔴 Needs Improvement"
            )
            st.markdown(f"**Overall RAG Quality:** {quality} (avg: {avg_score:.2f})")

            st.divider()
            st.markdown("###  Per-Question Breakdown")

            for i, item in enumerate(results["per_question"], 1):
                with st.expander(f"Q{i}: {item['question'][:80]}…"):
                    st.markdown(f"**Answer:** {item['answer']}")
                    st.markdown("**Scores:**")
                    s = item["scores"]
                    cols = st.columns(4)
                    cols[0].metric("Relevancy", f"{s['answer_relevancy']:.2f}")
                    cols[1].metric("Faithfulness", f"{s['faithfulness']:.2f}")
                    cols[2].metric("Precision", f"{s['context_precision']:.2f}")
                    cols[3].metric("Recall", f"{s['context_recall']:.2f}")
                    if item.get("reasoning"):
                        st.caption(f"💬 {item['reasoning']}")

            # Resume-ready summary
            st.divider()
            st.markdown("###  Resume-Ready Summary")
            st.code(
                f"Evaluated RAG pipeline using RAGAS-style LLM-as-judge — achieved "
                f"{metrics['faithfulness']:.2f} answer faithfulness and "
                f"{metrics['context_precision']:.2f} context precision across "
                f"{results['num_questions']} test queries",
                language=None,
            )
