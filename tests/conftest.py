import os
import pytest
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from local_dir_rag.embed import embed_docs


@pytest.fixture
def temp_dir():
    """Create a temporary directory fixture."""
    with TemporaryDirectory() as td:
        yield td


@pytest.fixture
def mock_openai_embeddings(monkeypatch):
    """Mock OpenAI embeddings to avoid API calls during tests."""
    mock_embed = MagicMock()

    # Return a predictable embedding vector
    mock_embed.embed_documents.return_value = [[0.1, 0.2, 0.3] * 341]
    mock_embed.embed_query.return_value = [0.1, 0.2, 0.3] * 341

    return mock_embed


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(
            page_content="This is a test document about artificial intelligence.",
            metadata={"source": "test_doc_1.txt", "page": 1}
        ),
        Document(
            page_content="RAG systems combine retrieval with generation for better answers.",
            metadata={"source": "test_doc_2.txt", "page": 1}
        ),
        Document(
            page_content="Vector databases store embeddings for efficient similarity search.",
            metadata={"source": "test_doc_3.txt", "page": 1}
        )
    ]


@pytest.fixture
def mock_faiss_db(sample_documents, mock_openai_embeddings):
    """Create a mock FAISS database with sample documents."""
    with TemporaryDirectory() as td:
        db_path = os.path.join(td, "vector_db")

        # Create a real FAISS database with our mock embeddings
        db = FAISS.from_documents(
            documents=sample_documents,
            embedding=mock_openai_embeddings
        )
        db.save_local(db_path)

        yield db_path


@pytest.fixture
def test_file_structure(temp_dir):
    """Create a test file structure with sample text files."""
    docs_dir = os.path.join(temp_dir, "docs")
    os.makedirs(docs_dir, exist_ok=True)

    # Create sample text files
    file_paths = []
    for i in range(3):
        file_path = os.path.join(docs_dir, f"test_doc_{i}.txt")
        with open(file_path, "w") as f:
            f.write(f"This is test document {i} with some content for testing.\n")
            f.write(f"It contains multiple sentences to test chunking.\n")
            f.write(f"Each document has unique content to identify it in searches.")
        file_paths.append(file_path)

    return docs_dir, file_paths
