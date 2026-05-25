"""Query a local FAISS vector database and return raw matched chunks."""
import os
import json
import logging
from typing import Any

from dotenv import load_dotenv
from langchain_core.documents import Document

from local_dir_rag.vector_store import load_vector_database

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s `%(funcName)s` %(levelname)s:\n  %(message)s"
)
logger = logging.getLogger(__name__)


def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Remove noisy metadata fields and shorten source file path."""
    clean_metadata = dict(metadata)
    clean_metadata.pop("producer", None)
    clean_metadata.pop("creator", None)
    clean_metadata.pop("creationdate", None)
    clean_metadata.pop("moddate", None)
    clean_metadata.pop("total_pages", None)
    if "source" in clean_metadata:
        clean_metadata["source"] = os.path.split(clean_metadata["source"])[1]
    return clean_metadata


def _embed_query(vector_db, query: str) -> list[float]:
    """Embed query text for vector similarity lookup."""
    embeddings_model = getattr(vector_db, "embeddings", None)
    if embeddings_model is not None and hasattr(embeddings_model, "embed_query"):
        return embeddings_model.embed_query(query)

    embedding_function = getattr(vector_db, "embedding_function", None)
    if embedding_function is not None:
        if hasattr(embedding_function, "embed_query"):
            return embedding_function.embed_query(query)
        if callable(embedding_function):
            return embedding_function(query)

    raise ValueError("Vector database does not provide query embeddings.")


def retrieve_raw_matches(
    query: str,
    vector_db,
    k: int = 30
) -> list[dict[str, Any]]:
    """Retrieve raw document chunks and metadata for a query."""
    query_embedding = _embed_query(vector_db, query)
    documents: list[Document] = vector_db.similarity_search_by_vector(
        query_embedding,
        k=k
    )

    return [
        {
            "content": doc.page_content,
            "metadata": _sanitize_metadata(doc.metadata),
        }
        for doc in documents
    ]


def query_vector_db(
    query: str,
    vector_db_path: str,
    k: int = 30
) -> list[dict[str, Any]]:
    """Load an existing vector database and return raw retrieval matches."""
    vector_db = load_vector_database(vector_db_path)
    if vector_db is None:
        raise ValueError(f"No vector database found at {vector_db_path}.")
    return retrieve_raw_matches(query=query, vector_db=vector_db, k=k)


def query_loop(vector_db_path=None, k: int = 30):
    """
    Run an interactive retrieval session using a local vector database.
    """
    vector_db = load_vector_database(vector_db_path)
    if vector_db is None:
        raise ValueError(f"No vector database found at {vector_db_path}.")
    logger.info("Vector database loaded successfully from %s", vector_db_path)

    print("Local RAG Retrieval Session")
    print("Type your questions below.")
    print("Type 'exit' or 'quit' to end the session.")

    while True:
        prompt = input("\nPrompt: ")

        # Check for exit command
        if prompt.lower() in ['exit', 'quit']:
            print("Exiting chat session.")
            break

        matches = retrieve_raw_matches(query=prompt, vector_db=vector_db, k=k)
        print("\nResponse:")
        print(json.dumps(matches, indent=2))


if __name__ == "__main__":
    # Load environment variables
    load_dotenv()

    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH")
    query_loop(VECTOR_DB_PATH)
