import os
import re
import time
import random
from langchain_groq import ChatGroq

MODEL_NAME = "openai/gpt-oss-20b"
MAX_RETRIES = 5
BASE_WAIT = 3

from dotenv import load_dotenv

load_dotenv()

def _get_api_key() -> str:
    """
    Resolve Groq API key from:
    1. Streamlit Cloud secrets (st.secrets) — used in deployment
    2. Local .env via python-dotenv — used locally
    """
    try:
        import streamlit as st
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY", "")


def _parse_suggested_wait(error_message: str):
    """Extract Groq's suggested wait time from error, e.g. 'try again in 33.97s'."""
    match = re.search(r"try again in ([\d.]+)s", error_message)
    if match:
        return float(match.group(1)) + 1.0
    return None


def invoke_with_retry(llm: ChatGroq, prompt: str, max_retries: int = MAX_RETRIES) -> str:
    """Invoke LLM with exponential backoff on 429 rate-limit errors."""
    for attempt in range(1, max_retries + 1):
        try:
            response = llm.invoke(prompt)
            return response.content
        except Exception as e:
            err = str(e)
            is_rate_limit = "429" in err or "rate_limit" in err.lower()
            if is_rate_limit and attempt < max_retries:
                wait = _parse_suggested_wait(err)
                if wait is None:
                    wait = BASE_WAIT * (2 ** (attempt - 1)) + random.uniform(0, 1)
                time.sleep(wait)
            else:
                raise
