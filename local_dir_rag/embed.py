"""Main entry point for the local-dir-rag package."""

import logging
import os
from typing import Iterable

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from local_dir_rag.document_loader import (
    get_files_from_directory,
    load_document,
)
from local_dir_rag.file_tracker import FileTracker
from local_dir_rag.text_processor import split_documents
from local_dir_rag.vector_store import (
    load_vector_database,
    remove_documents_by_source,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s `%(funcName)s` %(levelname)s:\n  %(message)s"
)
logger = logging.getLogger(__name__)


def _normalize_docs_paths(docs_paths: str | Iterable[str] | None) -> list[str]:
    """Normalize doc paths from string or iterable into a list."""
    if docs_paths is None:
        return []

    if isinstance(docs_paths, str):
        raw_paths = docs_paths.split(os.pathsep)
    else:
        raw_paths = list(docs_paths)

    normalized_paths = [
        path.strip()
        for path in raw_paths
        if path and path.strip()
    ]
    return normalized_paths


def embed_docs(
    docs_paths: str | Iterable[str] = None,
    vector_db_path: str = None,
    embeddings_model: Embeddings = None
):
    """
    Create and save a vector database from documents.

    Uses incremental indexing: only new or modified files are processed.
    Modified files have their old chunks removed before re-indexing.
    Deleted files have their chunks removed from the vector store.

    Args:
        docs_paths (str | Iterable[str], optional): One or more document
            directories. Strings may contain multiple paths separated by
            ``os.pathsep``.
        vector_db_path (str, optional): Path to save the vector database.
        embeddings_model (Embeddings, optional): Embedding model to use.

    Returns:
        FAISS: The vector database.
    """
    if embeddings_model is None:
        embeddings_model = OpenAIEmbeddings()

    normalized_docs_paths = _normalize_docs_paths(docs_paths)
    if len(normalized_docs_paths) == 0:
        raise ValueError("Documents path is not set.")

    # Initialize file tracker (creates directory if needed)
    file_tracker = FileTracker(vector_db_path)

    # Attempt to load vector database from the specified path,
    # if it exists
    vector_db = load_vector_database(
        vector_db_path,
        embeddings_model
    )
    logger.info("Vector database path %s", vector_db_path)

    files: list[str] = []
    for docs_directory in normalized_docs_paths:
        if not os.path.isdir(docs_directory):
            logger.error(
                "Documents path does not exist or is not a directory: %s",
                docs_directory
            )
            continue
        logger.info("Loading documents from %s", docs_directory)
        files.extend(get_files_from_directory(docs_directory))

    # Handle deleted files
    deleted_files = file_tracker.get_deleted_files(files)
    for deleted_file in deleted_files:
        logger.info("File deleted: %s", deleted_file)
        if vector_db is not None:
            remove_documents_by_source(vector_db, deleted_file)
        file_tracker.remove_file(deleted_file)

    # Process each file
    files_processed = 0
    files_skipped = 0

    for file_path in files:
        file_status = file_tracker.get_file_status(file_path)
        _, file_name = os.path.split(file_path)

        if not file_status.needs_indexing:
            logger.info("Skipping unchanged file: %s", file_name)
            files_skipped += 1
            continue

        # If file was modified, remove old chunks first
        if file_status.is_modified and vector_db is not None:
            logger.info("File modified, removing old chunks: %s", file_name)
            remove_documents_by_source(vector_db, file_path)

        documents = load_document(file_path)
        logger.info(
            "Loaded %d documents from '%s'",
            len(documents),
            file_name
        )
        chunks = split_documents(documents)
        if len(chunks) == 0:
            logger.warning("No chunks created from %s", file_name)
            continue
        logger.info("Created %d chunks", len(chunks))
        # Add chunks to the vector database
        if vector_db is None:
            vector_db = FAISS.from_documents(
                chunks,
                embeddings_model
            )
        else:
            # Append to the existing database
            vector_db.add_documents(chunks)
        logger.info("Added %d chunks to the database", len(chunks))
        vector_db.save_local(vector_db_path)

        # Update file tracker after successful indexing
        file_tracker.update_file_checksum(file_path)
        files_processed += 1

    logger.info(
        "Indexing complete: %d files processed, %d files skipped",
        files_processed,
        files_skipped
    )

    return vector_db


if __name__ == "__main__":
    _docs_paths = _normalize_docs_paths(os.getenv("DOCS_PATH"))
    if len(_docs_paths) == 0:
        raise ValueError("Documents path is not set.")

    _vector_db_path = os.getenv("VECTOR_DB_PATH")
    if _vector_db_path is None:
        raise ValueError("Vector database path is not set.")

    embed_docs(docs_paths=_docs_paths, vector_db_path=_vector_db_path)
