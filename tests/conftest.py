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
