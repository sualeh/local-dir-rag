"""Processor for splitting documents into chunks and formatting them."""
import os
import logging
import json
from langchain.schema import Document
from langchain.text_splitter import SentenceTransformersTokenTextSplitter

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(None)


def split_documents(
    documents: list[Document],
    chunk_size: int = 1024,  # in tokens, not characters
    chunk_overlap: int = 150  # ~10% overlap preserves context
) -> list[Document]:
    """
    Split documents into smaller chunks for better processing.

    Args:
        documents: List of Document objects to split.
        chunk_size: Maximum size of each chunk in characters.
        chunk_overlap: Number of characters of overlap between chunks.

    Returns:
        List of smaller Document chunks.
    """
    text_splitter = SentenceTransformersTokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    logger.info(
        "Split into %s documents into %d chunks.",
        len(documents),
        len(chunks)
    )
    return chunks


def format_documents(documents: list[Document]) -> str:
    """
    Format the retrieved documents into a single context string.

    Args:
        docs: List of document objects with page_content and metadata

    Returns:
        str: Formatted context string
    """
    return "\n\n".join(doc.page_content for doc in documents)


def print_sources(documents: list[Document]) -> str:
    """
    Logs and prints metadata and content summary of a list of documents.
    Args:
        documents (list[Document]): A list of Document objects to process.
            Each Document is expected to have
            `metadata` (a dictionary) and `page_content` (a string).
    Returns:
        str: The input list of documents.
    Logs:
        - The total number of documents retrieved.
        - Metadata and a truncated preview (first 50 and last 50 characters)
          of the `page_content` for each document.
    """
    logger.info("Retrieved %d documents:", len(documents))
    for i, doc in enumerate(documents):
        metadata = doc.metadata
        metadata.pop("producer", None)
        metadata.pop("creator", None)
        metadata.pop("creationdate", None)
        metadata.pop("moddate", None)
        metadata.pop("total_pages", None)
        if "source" in metadata:
            metadata["source"] = os.path.split(metadata["source"])[1]
        page_content = doc.page_content.replace("\n", "")
        print(f"Source [{i+1}]:\n"
              f"{json.dumps(metadata)}\n"
              f"{page_content[:50]} ... {page_content[-50:]}\n")

    return documents
