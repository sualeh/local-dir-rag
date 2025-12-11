"""Tests for incremental embedding behavior."""
import os
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from local_dir_rag.embed import embed_docs
from local_dir_rag.file_tracker import FileTracker
from local_dir_rag.vector_store import remove_documents_by_source


class MockEmbeddings(Embeddings):
    """Mock embeddings model for testing."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return mock embeddings."""
        return [[0.1, 0.2, 0.3] * 128 for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        """Return mock embedding for a query."""
        return [0.1, 0.2, 0.3] * 128


@pytest.fixture
def docs_and_vector_db(temp_dir):
    """Create docs directory and vector db path."""
    docs_dir = os.path.join(temp_dir, "docs")
    vector_db_path = os.path.join(temp_dir, "vector_db")
    os.makedirs(docs_dir, exist_ok=True)
    return docs_dir, vector_db_path


def test_embed_new_files(docs_and_vector_db):
    """Test embedding new files creates tracker entries."""
    docs_dir, vector_db_path = docs_and_vector_db
    mock_embeddings = MockEmbeddings()

    # Create test files
    file1 = os.path.join(docs_dir, "file1.txt")
    file2 = os.path.join(docs_dir, "file2.txt")

    with open(file1, "w", encoding="utf-8") as f:
        f.write("Content for file 1. " * 10)
    with open(file2, "w", encoding="utf-8") as f:
        f.write("Content for file 2. " * 10)

    # Run embedding
    vector_db = embed_docs(
        docs_directory=docs_dir,
        vector_db_path=vector_db_path,
        embeddings_model=mock_embeddings
    )

    assert vector_db is not None

    # Check that tracker has entries
    tracker = FileTracker(vector_db_path)
    tracked_files = tracker.get_all_tracked_files()
    assert len(tracked_files) == 2
    assert file1 in tracked_files
    assert file2 in tracked_files


def test_skip_unchanged_files(docs_and_vector_db):
    """Test that unchanged files are skipped on re-run."""
    docs_dir, vector_db_path = docs_and_vector_db
    mock_embeddings = MockEmbeddings()

    # Create test file
    file1 = os.path.join(docs_dir, "file1.txt")
    with open(file1, "w", encoding="utf-8") as f:
        f.write("Content for file 1. " * 10)

    # First run
    vector_db1 = embed_docs(
        docs_directory=docs_dir,
        vector_db_path=vector_db_path,
        embeddings_model=mock_embeddings
    )
    initial_count = len(vector_db1.index_to_docstore_id)

    # Second run without changes
    vector_db2 = embed_docs(
        docs_directory=docs_dir,
        vector_db_path=vector_db_path,
        embeddings_model=mock_embeddings
    )

    # Count should remain the same (no duplicates)
    final_count = len(vector_db2.index_to_docstore_id)
    assert final_count == initial_count


def test_reindex_modified_files(docs_and_vector_db):
    """Test that modified files are re-indexed with old chunks removed."""
    docs_dir, vector_db_path = docs_and_vector_db
    mock_embeddings = MockEmbeddings()

    # Create test file
    file1 = os.path.join(docs_dir, "file1.txt")
    with open(file1, "w", encoding="utf-8") as f:
        f.write("Original content. " * 10)

    # First run
    vector_db1 = embed_docs(
        docs_directory=docs_dir,
        vector_db_path=vector_db_path,
        embeddings_model=mock_embeddings
    )
    initial_count = len(vector_db1.index_to_docstore_id)

    # Modify the file
    with open(file1, "w", encoding="utf-8") as f:
        f.write("Modified content. " * 10)

    # Second run
    vector_db2 = embed_docs(
        docs_directory=docs_dir,
        vector_db_path=vector_db_path,
        embeddings_model=mock_embeddings
    )

    # Count should remain the same (old chunks removed, new ones added)
    final_count = len(vector_db2.index_to_docstore_id)
    assert final_count == initial_count

    # Verify the content is from the modified file
    # by checking that at least one document contains "Modified"
    found_modified = False
    for doc_id in vector_db2.index_to_docstore_id.values():
        doc = vector_db2.docstore.search(doc_id)
        if "Modified" in doc.page_content:
            found_modified = True
            break
    assert found_modified


def test_add_new_file_incrementally(docs_and_vector_db):
    """Test adding a new file to an existing index."""
    docs_dir, vector_db_path = docs_and_vector_db
    mock_embeddings = MockEmbeddings()

    # Create first file
    file1 = os.path.join(docs_dir, "file1.txt")
    with open(file1, "w", encoding="utf-8") as f:
        f.write("Content for file 1. " * 10)

    # First run
    vector_db1 = embed_docs(
        docs_directory=docs_dir,
        vector_db_path=vector_db_path,
        embeddings_model=mock_embeddings
    )
    initial_count = len(vector_db1.index_to_docstore_id)

    # Add second file
    file2 = os.path.join(docs_dir, "file2.txt")
    with open(file2, "w", encoding="utf-8") as f:
        f.write("Content for file 2. " * 10)

    # Second run
    vector_db2 = embed_docs(
        docs_directory=docs_dir,
        vector_db_path=vector_db_path,
        embeddings_model=mock_embeddings
    )

    # Count should increase
    final_count = len(vector_db2.index_to_docstore_id)
    assert final_count > initial_count

    # Both files should be tracked
    tracker = FileTracker(vector_db_path)
    tracked_files = tracker.get_all_tracked_files()
    assert len(tracked_files) == 2


def test_handle_deleted_files(docs_and_vector_db):
    """Test that deleted files have their chunks removed."""
    docs_dir, vector_db_path = docs_and_vector_db
    mock_embeddings = MockEmbeddings()

    # Create two files
    file1 = os.path.join(docs_dir, "file1.txt")
    file2 = os.path.join(docs_dir, "file2.txt")

    with open(file1, "w", encoding="utf-8") as f:
        f.write("Content for file 1. " * 10)
    with open(file2, "w", encoding="utf-8") as f:
        f.write("Content for file 2. " * 10)

    # First run
    vector_db1 = embed_docs(
        docs_directory=docs_dir,
        vector_db_path=vector_db_path,
        embeddings_model=mock_embeddings
    )
    initial_count = len(vector_db1.index_to_docstore_id)

    # Delete file2
    os.remove(file2)

    # Second run
    vector_db2 = embed_docs(
        docs_directory=docs_dir,
        vector_db_path=vector_db_path,
        embeddings_model=mock_embeddings
    )

    # Count should decrease
    final_count = len(vector_db2.index_to_docstore_id)
    assert final_count < initial_count

    # Only file1 should be tracked
    tracker = FileTracker(vector_db_path)
    tracked_files = tracker.get_all_tracked_files()
    assert len(tracked_files) == 1
    assert file1 in tracked_files
    assert file2 not in tracked_files


def test_sqlite_db_location(docs_and_vector_db):
    """Test that SQLite database is created in vector_db_path."""
    docs_dir, vector_db_path = docs_and_vector_db
    mock_embeddings = MockEmbeddings()

    # Create test file
    file1 = os.path.join(docs_dir, "file1.txt")
    with open(file1, "w", encoding="utf-8") as f:
        f.write("Content for file 1. " * 10)

    # Run embedding
    embed_docs(
        docs_directory=docs_dir,
        vector_db_path=vector_db_path,
        embeddings_model=mock_embeddings
    )

    # Check SQLite database exists in vector_db_path
    db_path = os.path.join(vector_db_path, "file_tracker.db")
    assert os.path.exists(db_path)


def test_remove_documents_by_source():
    """Test removing documents by source from vector store."""
    # Create a mock FAISS vector store
    mock_vector_db = MagicMock()

    # Setup mock docstore with documents
    doc1 = Document(
        page_content="Content 1",
        metadata={"source": "/path/to/file1.txt"}
    )
    doc2 = Document(
        page_content="Content 2",
        metadata={"source": "/path/to/file2.txt"}
    )
    doc3 = Document(
        page_content="Content 3",
        metadata={"source": "/path/to/file1.txt"}
    )

    mock_vector_db.index_to_docstore_id = {0: "id1", 1: "id2", 2: "id3"}
    mock_vector_db.docstore.search = MagicMock(
        side_effect=lambda doc_id: {
            "id1": doc1,
            "id2": doc2,
            "id3": doc3
        }[doc_id]
    )

    # Remove documents for file1
    removed = remove_documents_by_source(
        mock_vector_db,
        "/path/to/file1.txt"
    )

    # Should have found 2 documents to remove
    assert removed == 2
    mock_vector_db.delete.assert_called_once()
    # Verify the correct IDs were passed to delete
    call_args = mock_vector_db.delete.call_args[0][0]
    assert "id1" in call_args
    assert "id3" in call_args
    assert "id2" not in call_args


def test_remove_documents_from_none_db():
    """Test that removing from None database returns 0."""
    removed = remove_documents_by_source(None, "/path/to/file.txt")
    assert removed == 0
