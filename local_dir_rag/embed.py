"""Main entry point for the local-dir-rag package."""

import os
import logging
from langchain.embeddings.base import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from local_dir_rag.document_loader import get_files_from_directory, load_document
from local_dir_rag.text_processor import split_documents

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(None)


def embed(
    docs_directory: str = None,
    vector_db_path: str = None,
    embeddings_model: Embeddings = None
):
    """
    Create and save a vector database from documents.

    Args:
        docs_directory (str, optional): Directory containing documents to
            process.
        vector_db_path (str, optional): Path to save the vector database.
    """
    if embeddings_model is None:
        embeddings_model = OpenAIEmbeddings()
    vector_db = None
    logger.info("Vector database saved to %s", vector_db_path)

    logger.info("Loading documents from %s", docs_directory)
    files = get_files_from_directory(docs_directory)

    for file_path in files:
        documents = load_document(file_path)
        _, file_name = os.path.split(file_path)
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

    return vector_db


if __name__ == "__main__":
    _docs_directory = os.getenv("DOCS_DIRECTORY")
    if _docs_directory is None or not os.path.exists(_docs_directory):
        raise ValueError("Documents directory is not set.")

    _vector_db_path = os.getenv("VECTOR_DB_PATH")
    if _vector_db_path is None or not os.path.exists(_vector_db_path):
        raise ValueError("Vector database path is not set.")

    embed(docs_directory=_docs_directory, vector_db_path=_vector_db_path)
