"""Main entry point for the local-dir-rag package."""

import logging
import os

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


def embed_docs(
    docs_directory: str = None,
    vector_db_path: str = None,
    embeddings_model: Embeddings = None
):
    """
    Create and save a vector database from documents.

    Uses incremental indexing: only new or modified files are processed.
    Modified files have their old chunks removed before re-indexing.
    Deleted files have their chunks removed from the vector store.

    Args:
        docs_directory (str, optional): Directory containing documents to
            process.
        vector_db_path (str, optional): Path to save the vector database.
        embeddings_model (Embeddings, optional): Embedding model to use.

    Returns:
        FAISS: The vector database.
    """
    if embeddings_model is None:
        embeddings_model = OpenAIEmbeddings()

    # Initialize file tracker (creates directory if needed)
    file_tracker = FileTracker(vector_db_path)

    # Attempt to load vector database from the specified path,
    # if it exists
    vector_db = load_vector_database(
        vector_db_path,
        embeddings_model
    )
    logger.info("Vector database path %s", vector_db_path)

    logger.info("Loading documents from %s", docs_directory)
    files = get_files_from_directory(docs_directory)

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
    _docs_directory = os.getenv("DOCS_DIRECTORY")
    if _docs_directory is None or not os.path.exists(_docs_directory):
        raise ValueError("Documents directory is not set.")

    _vector_db_path = os.getenv("VECTOR_DB_PATH")
    if _vector_db_path is None or not os.path.exists(_vector_db_path):
        raise ValueError("Vector database path is not set.")

    embed_docs(docs_directory=_docs_directory, vector_db_path=_vector_db_path)
