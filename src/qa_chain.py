from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
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

Question: {input}

Answer:"""


def _format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_qa_chain():
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2,
        max_tokens=512,
        api_key=_get_api_key(),
    )

    prompt = ChatPromptTemplate.from_template(QA_SYSTEM_PROMPT)

    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 6},
    )

    def retrieve_and_answer(input_dict):
        question = input_dict["input"]
        chat_history = input_dict.get("chat_history", "")
        docs = retriever.invoke(question)
        context = _format_docs(docs)

        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({
            "context": context,
            "chat_history": chat_history,
            "input": question,
        })

        return {"answer": answer, "context": docs}

    return retrieve_and_answer