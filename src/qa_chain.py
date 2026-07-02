from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from src.retriever import load_vectorstore
from src.llm import _get_api_key

QA_SYSTEM_PROMPT = """You are an expert research assistant.
Answer questions strictly based on the provided research paper excerpts.

Rules:
- Be thorough and detailed in your answers.
- Quote or closely reference specific parts of the papers when relevant.
- If the answer spans multiple papers, clearly attribute each point.
- If the context does not contain the answer, say so explicitly.

Context from papers:
{context}

Chat history:
{chat_history}
"""

def build_qa_chain():
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2,
        max_tokens=512,
        api_key=_get_api_key(),
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", QA_SYSTEM_PROMPT),
        ("human", "{input}"),
    ])

    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 6},
    )

    combine_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, combine_chain)