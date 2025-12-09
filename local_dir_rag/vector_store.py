"""Utilities for loading and managing the local FAISS vector store."""

import os
import logging
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s `%(funcName)s` %(levelname)s:\n  %(message)s"
)
logger = logging.getLogger(__name__)


def remove_documents_by_source(vector_db: FAISS, source_path: str) -> int:
    """
    Remove all documents from the vector store that match the given source.

    Args:
        vector_db: The FAISS vector database.
        source_path: The source file path to match in document metadata.

    Returns:
        Number of documents removed.
    """
    if vector_db is None:
        return 0

    # Get all document IDs that match the source
    ids_to_remove = []
    docstore = vector_db.docstore

    # FAISS uses index_to_docstore_id to map internal indices to doc IDs
    for doc_id in vector_db.index_to_docstore_id.values():
        doc = docstore.search(doc_id)
        if doc and doc.metadata.get("source") == source_path:
            ids_to_remove.append(doc_id)

    if ids_to_remove:
        vector_db.delete(ids_to_remove)
        logger.info(
            "Removed %d chunks for source %s",
            len(ids_to_remove),
            source_path
        )

    return len(ids_to_remove)


def load_vector_database(
    db_path,
    embeddings_model: Embeddings = None
) -> FAISS:
    """
    Load a FAISS vector database from the specified path.

    Args:
        db_path (str): Path to the vector database
        embeddings: Embedding model to use (default: OpenAIEmbeddings)

    Returns:
        FAISS: The loaded vector database or None if not found
    """
    if embeddings_model is None:
        embeddings_model = OpenAIEmbeddings()

    # Check if the FAISS index file exists (not just the directory)
    index_file = os.path.join(db_path, "index.faiss")
    if not os.path.exists(index_file):
        logger.info("No existing vector database found at %s", db_path)
        return None

    try:
        vector_db = FAISS.load_local(
            db_path,
            embeddings_model,
            allow_dangerous_deserialization=True
        )
        logger.info("Vector database successfully loaded from %s", db_path)
        return vector_db
    except (FileNotFoundError, OSError, ValueError) as error:
        logger.error(
            "Error loading vector database from %s: %s",
            db_path,
            error,
        )
        return None
