import os
from tempfile import TemporaryDirectory

import pytest
from langchain_core.documents import Document

# pylint: disable=redefined-outer-name


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
            page_content=(
                "This is a test document about artificial "
                "intelligence."
            ),
            metadata={"source": "test_doc_1.txt", "page": 1}
        ),
        Document(
            page_content=(
                "RAG systems combine retrieval with generation for "
                "better answers."
            ),
            metadata={"source": "test_doc_2.txt", "page": 1}
        ),
        Document(
            page_content=(
                "Vector databases store embeddings for efficient "
                "similarity search."
            ),
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
        with open(file_path, "w", encoding="utf-8") as file_handle:
            file_handle.write(
                f"This is test document {i} with some content for testing.\n"
            )
            file_handle.write(
                "It contains multiple sentences to test chunking.\n"
            )
            file_handle.write(
                "Each document has unique content to identify it in searches."
            )
        file_paths.append(file_path)

    return docs_dir, file_paths


@pytest.fixture
def test_file_structure_with_subdirs(temp_dir):
    """Create a test file structure with subdirectories."""
    docs_dir = os.path.join(temp_dir, "docs")
    os.makedirs(docs_dir, exist_ok=True)

    # Create subdirectories
    subdir1 = os.path.join(docs_dir, "subdir1")
    subdir2 = os.path.join(docs_dir, "subdir2")
    nested_subdir = os.path.join(subdir1, "nested")
    os.makedirs(subdir1, exist_ok=True)
    os.makedirs(subdir2, exist_ok=True)
    os.makedirs(nested_subdir, exist_ok=True)

    file_paths = []

    # Create files in root docs directory
    for i in range(2):
        file_path = os.path.join(docs_dir, f"root_doc_{i}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"Root test document {i}")
        file_paths.append(file_path)

    # Create files in subdir1
    for i in range(2):
        file_path = os.path.join(subdir1, f"subdir1_doc_{i}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"Subdir1 test document {i}")
        file_paths.append(file_path)

    # Create files in subdir2
    file_path = os.path.join(subdir2, "subdir2_doc.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("Subdir2 test document")
    file_paths.append(file_path)

    # Create file in nested subdirectory
    file_path = os.path.join(nested_subdir, "nested_doc.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("Nested test document")
    file_paths.append(file_path)

    return docs_dir, file_paths
