from langchain_core.documents import Document


def format_citations(context_docs: list[Document]) -> list[str]:
    """
    Format retrieved document chunks into readable citation strings.
    Deduplicates by (paper, page).
    """
    seen = set()
    citations = []

    for doc in context_docs:
        paper = doc.metadata.get("paper", "Unknown Paper")
        page = doc.metadata.get("page", None)

        key = (paper, page)
        if key in seen:
            continue
        seen.add(key)

        if page is not None:
            citations.append(f"**{paper}** — Page {int(page) + 1}")
        else:
            citations.append(f"**{paper}**")

    return citations
