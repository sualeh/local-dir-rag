import os
import logging
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s"
)


def create_vector_db(
    chunks: list[Document],
    embeddings_model: Embeddings = None,
    save_path: str = None
) -> FAISS:
    """
    Create a vector database from document chunks.

    Args:
        chunks: List of Document chunks to store in the database.
        embeddings_model: Model to create vector embeddings from text.
        save_path: Optional path to save the vector database.

    Returns:
        FAISS vector database containing the document embeddings.
    """
    if embeddings_model is None:
        embeddings_model = OpenAIEmbeddings()

    vector_db = FAISS.from_documents(chunks, embeddings_model)

    if save_path:
        vector_db.save_local(save_path)
        logger.info("Vector database saved to %s", save_path)

    return vector_db


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

    if not os.path.exists(db_path):
        logger.error("Error: Vector database not found at %s", db_path)
        return None

    try:
        vector_db = FAISS.load_local(
            db_path,
            embeddings_model,
            allow_dangerous_deserialization=True
        )
        logger.info("Vector database successfully loaded from %s", db_path)
        return vector_db
    except Exception as e:
        logger.error("Error loading vector database: %s", e)
        return None


def query_vector_db(
    query: str,
    vector_db: FAISS,
    k: int = 5
) -> list[tuple[Document, float]]:
    """
    Query the vector database for similar documents.

    Args:
        query: Query string to search for.
        vector_db: FAISS vector database to search in.
        k: Number of results to return.

    Returns:
        List of tuples containing (Document, similarity_score).
    """
    results = vector_db.similarity_search_with_score(query, k=k)

    return results
