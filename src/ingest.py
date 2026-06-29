import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_documents(pdf_paths: list[str]) -> list:
    """Load PDF documents and tag each chunk with its paper name."""
    all_docs = []

    for path in pdf_paths:
        loader = PyPDFLoader(path)
        docs = loader.load()

        paper_name = Path(path).stem

        for doc in docs:
            doc.metadata["paper"] = paper_name
            doc.metadata["source"] = path

        all_docs.extend(docs)

    return all_docs


def split_documents(documents: list) -> list:
    """
    Split documents into chunks.
    Larger chunks (1500 tokens) with good overlap preserve context
    for summaries and QA.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    # Preserve paper metadata on every chunk
    for chunk in chunks:
        if "paper" not in chunk.metadata:
            chunk.metadata["paper"] = "Unknown Paper"

    return chunks
