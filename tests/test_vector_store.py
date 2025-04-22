import os
import pytest
from unittest.mock import patch
from local_dir_rag.vector_store import load_vector_database


def test_load_vector_database(mock_faiss_db, mock_openai_embeddings):
    """Test loading a vector database from a path."""
    # Load the database
    db = load_vector_database(mock_faiss_db, mock_openai_embeddings)
    assert db is not None

    # Test querying the database
    results = db.similarity_search("artificial intelligence", k=1)
    assert len(results) == 1


def test_load_vector_database_nonexistent_path(temp_dir):
    """Test loading a vector database from a nonexistent path."""
    nonexistent_path = os.path.join(temp_dir, "nonexistent")
    db = load_vector_database(nonexistent_path)
    assert db is None


def test_load_vector_database_exception_handling(temp_dir, mock_openai_embeddings):
    """Test handling exceptions when loading a vector database."""
    # Create an invalid "database" (just a file, not a FAISS index)
    invalid_db_path = os.path.join(temp_dir, "invalid_db")
    os.makedirs(invalid_db_path, exist_ok=True)
    with open(os.path.join(invalid_db_path, "dummy_file"), "w") as f:
        f.write("This is not a FAISS index")

    # Test loading the invalid database
    with patch("logging.error") as mock_log:
        db = load_vector_database(invalid_db_path, mock_openai_embeddings)
        assert db is None
        mock_log.assert_called_once()
