"""Tests for query-time retrieval behavior."""
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from local_dir_rag.query_with_rag import query_vector_db, retrieve_raw_matches


def test_query_vector_db_does_not_index_documents():
    """Query path should load and search only, without indexing operations."""
    mock_vector_db = MagicMock()
    mock_vector_db.embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
    mock_vector_db.similarity_search_by_vector.return_value = []

    with patch(
        "local_dir_rag.query_with_rag.load_vector_database",
        return_value=mock_vector_db
    ) as mock_loader:
        results = query_vector_db(
            query="find content",
            vector_db_path="/tmp/vector_db",
            k=5
        )

    assert results == []
    mock_loader.assert_called_once_with("/tmp/vector_db")
    mock_vector_db.add_documents.assert_not_called()
    mock_vector_db.save_local.assert_not_called()


def test_retrieve_raw_matches_uses_query_embedding_for_lookup():
    """The incoming query text should be embedded for similarity search."""
    mock_vector_db = MagicMock()
    mock_vector_db.embeddings.embed_query.return_value = [1.0, 2.0, 3.0]
    mock_vector_db.similarity_search_by_vector.return_value = [
        Document(
            page_content="Chunk content",
            metadata={"source": "/tmp/docs/example.txt", "page": 1}
        )
    ]

    retrieve_raw_matches(query="sample query", vector_db=mock_vector_db, k=7)

    mock_vector_db.embeddings.embed_query.assert_called_once_with(
        "sample query"
    )
    mock_vector_db.similarity_search_by_vector.assert_called_once_with(
        [1.0, 2.0, 3.0],
        k=7
    )


def test_retrieve_raw_matches_returns_content_and_metadata():
    """Return raw matched chunk text with cleaned metadata."""
    mock_vector_db = MagicMock()
    mock_vector_db.embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
    mock_vector_db.similarity_search_by_vector.return_value = [
        Document(
            page_content="Retrieved chunk body",
            metadata={
                "source": "/tmp/docs/file.txt",
                "page": 4,
                "section": "overview",
                "producer": "ignored",
                "creator": "ignored",
            }
        )
    ]

    results = retrieve_raw_matches(
        query="show me overview",
        vector_db=mock_vector_db
    )

    assert len(results) == 1
    assert results[0]["content"] == "Retrieved chunk body"
    assert results[0]["metadata"] == {
        "source": "file.txt",
        "page": 4,
        "section": "overview",
    }
