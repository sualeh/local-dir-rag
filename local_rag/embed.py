"""Main entry point for the local-rag package."""

import os
import logging
from dotenv import load_dotenv
from local_rag.document_loader import load_documents_from_directory
from local_rag.text_processor import split_documents
from local_rag.vector_store import create_vector_db

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def embed(docs_directory: str = None, vector_db_path: str = None):
    """
    Create and save a vector database from documents.

    Args:
        docs_directory (str, optional): Directory containing documents to
            process.
        vector_db_path (str, optional): Path to save the vector database.
    """
    logger.info("Loading documents from %s", docs_directory)
    documents = load_documents_from_directory(docs_directory)
    logger.info("Loaded %d documents", len(documents))

    chunks = split_documents(documents)
    logger.info("Created %d chunks", len(chunks))

    logger.info("Creating vector database at %s", vector_db_path)
    vector_db = create_vector_db(
        chunks,
        save_path=vector_db_path
    )
    logger.info("Vector database created successfully at %s", vector_db_path)

    return vector_db


if __name__ == "__main__":
    docs_directory = os.getenv("DOCS_DIRECTORY")
    if docs_directory is None:
        raise ValueError("Documents directory is not set.")

    vector_db_path = os.getenv("VECTOR_DB_PATH")
    if vector_db_path is None:
        raise ValueError("Vector database path is not set.")

    embed(docs_directory=docs_directory, vector_db_path=vector_db_path)
