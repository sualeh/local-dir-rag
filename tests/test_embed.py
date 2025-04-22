import os
import pytest
from unittest.mock import patch, MagicMock
from local_dir_rag.embed import embed_docs


def test_embed_docs(test_file_structure, mock_openai_embeddings, temp_dir):
    """Test embedding documents from a directory."""
    docs_dir, _ = test_file_structure
    vector_db_path = os.path.join(temp_dir, "vector_db")

    # Test embedding docs
    db = embed_docs(
        docs_directory=docs_dir,
        vector_db_path=vector_db_path,
        embeddings_model=mock_openai_embeddings
    )

    # Check database was created
    assert db is not None
    assert os.path.exists(vector_db_path)
    assert os.path.exists(os.path.join(vector_db_path, "index.faiss"))

    # Check we can search the database
    results = db.similarity_search("test document", k=1)
    assert len(results) > 0


def test_embed_docs_incremental(test_file_structure, mock_openai_embeddings, temp_dir):
    """Test incremental addition of documents to an existing vector database."""
    docs_dir, file_paths = test_file_structure
    vector_db_path = os.path.join(temp_dir, "vector_db")

    # Create initial database with first document
    initial_dir = os.path.join(temp_dir, "initial_docs")
    os.makedirs(initial_dir, exist_ok=True)
    initial_file = os.path.join(initial_dir, "initial.txt")
    with open(initial_file, "w", encoding="utf-8") as f:
        f.write("This is an initial document for testing incremental updates.")

    # Create initial database
    db1 = embed_docs(
        docs_directory=initial_dir,
        vector_db_path=vector_db_path,
        embeddings_model=mock_openai_embeddings
    )

    # Now add more documents
    db2 = embed_docs(
        docs_directory=docs_dir,
        vector_db_path=vector_db_path,
        embeddings_model=mock_openai_embeddings
    )

    # Check database was updated
    assert db2 is not None

    # This should find both the initial document and the test documents
    results = db2.similarity_search("document", k=4)
    assert len(results) >= 3  # At least the initial doc and some test docs
