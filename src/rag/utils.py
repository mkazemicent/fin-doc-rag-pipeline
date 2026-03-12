"""Utility functions extracted from the Streamlit UI layer for testability."""
from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.rag.deal_analyzer import DealExtraction, IRRELEVANT_QUERY_TOKEN


def evaluate_show_transparency(
    routing_signal: str, optimized_query: str, chunks: list
) -> bool:
    """Decide whether to render the retrieval transparency expander."""
    return (
        routing_signal == "relevant"
        and optimized_query != IRRELEVANT_QUERY_TOKEN
        and len(chunks) > 0
    )


def serialize_for_history(content) -> str:
    """Convert a DealExtraction (or plain string) to a human-readable string for chat history."""
    if isinstance(content, DealExtraction):
        extraction = content
        return (
            f"Maturity Date: {extraction.maturity_date}. "
            f"Deal Terms: {', '.join(extraction.deal_terms)}. "
            f"Risk Factors: {', '.join(extraction.risk_factors)}."
        )
    return str(content)


def should_render_transparency(message: dict) -> bool:
    """Decide whether a stored message should render its transparency section on replay."""
    return (
        "transparency" in message
        and message.get("routing_signal") == "relevant"
    )


_DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def size_cap_chunk(
    semantic_texts: list[str],
    metadata: dict,
    max_chunk_size: int = 1500,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    separators: Optional[List[str]] = None,
) -> list[Document]:
    """Apply size-cap fallback to pre-split semantic text sections.

    Args:
        semantic_texts: Pre-split text sections (e.g. from SemanticChunker).
        metadata: Metadata dict to attach to each chunk (copied per chunk).
        max_chunk_size: Sections longer than this get re-split.
        chunk_size: Target chunk size for RecursiveCharacterTextSplitter.
        chunk_overlap: Overlap between sub-chunks.
        separators: Custom separator list. Defaults to ``["\\n\\n", "\\n", ". ", " ", ""]``.
    """
    if separators is None:
        separators = _DEFAULT_SEPARATORS
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
    )
    chunked: list[Document] = []
    for text in semantic_texts:
        if len(text) > max_chunk_size:
            sub_docs = size_splitter.split_documents(
                [Document(page_content=text, metadata=metadata.copy())]
            )
            chunked.extend(sub_docs)
        elif text.strip():
            chunked.append(Document(page_content=text, metadata=metadata.copy()))
    return chunked
