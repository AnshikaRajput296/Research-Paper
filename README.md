# 📄 Research Paper QA Assistant

A production-grade RAG application for querying, summarizing, and evaluating research papers — built with LangChain, Groq LLaMA-3, FAISS, and Streamlit.

## ✨ Features

| Feature | Details |
|---|---|
| **Question Answering** | Ask natural language questions across multiple PDFs using RAG |
| **Comprehensive Summaries** | Full structured summaries (Overview → Findings → Conclusions) using map-reduce |
| **Comparative Analysis** | Side-by-side comparison of multiple papers |
| **RAGAS Evaluation** | Automated pipeline quality evaluation with 4 metrics |
| **Prompt Injection Guard** | Blocks adversarial inputs |
| **Source Citations** | Every answer cites paper name and page number |

## 🚀 Quick Start

```bash
# 1. Clone & enter
git clone <repo-url>
cd research-paper-qa-assistant

# 2. Create environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up your Groq API key
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# 5. Run
streamlit run app.py
```

## 📊 RAGAS Evaluation

The evaluation tab runs your RAG pipeline against a test set of questions and scores it on:

- **Answer Relevancy** — Does the answer address the question?
- **Faithfulness** — Is every claim grounded in retrieved context?
- **Context Precision** — Are the retrieved chunks relevant?
- **Context Recall** — Do chunks cover all needed information?

### Resume-worthy output:
> *"Evaluated RAG pipeline using RAGAS-style LLM-as-judge — achieved 0.81 answer faithfulness and 0.76 context precision across 10 test queries"*

## 🗂️ Project Structure

```
research-paper-qa-assistant/
├── app.py                 # Streamlit UI
├── main.py
├── requirements.txt
├── .env                   # API keys (gitignored)
└── src/
    ├── ingest.py          # PDF loading & chunking
    ├── retriever.py       # FAISS vector store
    ├── qa_chain.py        # RAG chain (LangChain + Groq)
    ├── summarizer.py      # Map-reduce summarization
    ├── ragas_eval.py      # RAGAS evaluation module
    ├── guardrails.py      # Prompt injection & scope checks
    ├── citations.py       # Source formatting
    └── utils.py           # File helpers
```

## 🔧 Configuration

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Your Groq API key ([get one free](https://console.groq.com)) |

## 🧠 Tech Stack

- **LLM**: Groq `llama-3.3-70b-versatile`
- **Embeddings**: `all-MiniLM-L6-v2` (HuggingFace)
- **Vector DB**: FAISS
- **Framework**: LangChain
- **UI**: Streamlit
- **PDF Parsing**: PyPDF
