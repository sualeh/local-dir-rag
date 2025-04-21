"""Processor for splitting documents into chunks and formatting them."""
import os
import logging
import json
from langchain.schema import Document
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s `%(funcName)s` %(levelname)s:\n  %(message)s"
)
logger = logging.getLogger(__name__)


def recursive_character_splitter(chunk_size, chunk_overlap):
    """
    Creates a RecursiveCharacterTextSplitter for document chunking.

    Args:
        chunk_size: Maximum size of each chunk in characters.
        chunk_overlap: Number of characters of overlap between chunks.

    Returns:
        RecursiveCharacterTextSplitter:
            A text splitter configured with the specified parameters.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )


def sentence_splitter(chunk_size, chunk_overlap):
    """
    Creates a SentenceTransformersTokenTextSplitter for token-based
    document chunking.

    Args:
        chunk_size: Maximum size of each chunk in tokens.
        chunk_overlap: Number of tokens of overlap between chunks.

    Returns:
        SentenceTransformersTokenTextSplitter:
            A token-based text splitter configured with the specified
            parameters.
    """
    return SentenceTransformersTokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )


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
    text_splitter = recursive_character_splitter(
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
        page_content = doc.page_content.replace("\n", " ")
        print(
            f"Source [{i+1}]:\n"
            f"{json.dumps(metadata)}\n"
            f"{page_content[:100]}"
            "\n<... skipped ...>\n"
            f"{page_content[-100:]}\n"
        )

    return documents
