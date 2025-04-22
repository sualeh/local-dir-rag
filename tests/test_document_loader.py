import os
import pytest
from local_dir_rag.document_loader import get_files_from_directory, load_document


def test_get_files_from_directory(test_file_structure):
    """Test getting files from a directory with specific extensions."""
    docs_dir, expected_files = test_file_structure

    # Test with default extensions
    files = get_files_from_directory(docs_dir)
    assert len(files) == 3

    # Test with specific extension
    files = get_files_from_directory(docs_dir, extensions=[".txt"])
    assert len(files) == 3

    # Test with non-matching extension
    files = get_files_from_directory(docs_dir, extensions=[".pdf"])
    assert len(files) == 0


def test_load_document(test_file_structure):
    """Test loading a document from file path."""
    _, file_paths = test_file_structure

    for file_path in file_paths:
        docs = load_document(file_path)
        assert len(docs) == 1
        assert docs[0].page_content is not None
        assert len(docs[0].page_content) > 0
        assert "test document" in docs[0].page_content
        assert docs[0].metadata["source"] == file_path


def test_load_document_nonexistent_file():
    """Test loading a document that doesn't exist."""
    docs = load_document("nonexistent_file.txt")
    assert len(docs) == 0


def test_load_document_unsupported_format(temp_dir):
    """Test loading a document with unsupported format."""
    file_path = os.path.join(temp_dir, "test.unsupported")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("Test content")

    docs = load_document(file_path)
    assert len(docs) == 0
