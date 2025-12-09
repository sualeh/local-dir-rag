# pylint: disable=protected-access
from langchain_core.documents import Document
from local_dir_rag.text_processor import (
    split_documents,
    format_documents,
    recursive_character_splitter,
    sentence_splitter
)


def test_recursive_character_splitter():
    """Test creating a recursive character splitter."""
    splitter = recursive_character_splitter(100, 20)
    assert splitter._chunk_size == 100
    assert splitter._chunk_overlap == 20


def test_sentence_splitter():
    """Test creating a sentence transformer splitter."""
    splitter = sentence_splitter(100, 20)
    assert splitter._chunk_size == 100
    assert splitter._chunk_overlap == 20


def test_split_documents():
    """Test splitting documents into chunks."""
    # Create a test document with content that should split
    long_text = "This is sentence one. " * 20 + "This is sentence two. " * 20
    docs = [Document(page_content=long_text, metadata={"source": "test.txt"})]

    # Split with small chunk size
    chunks = split_documents(docs, chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1

    # Ensure metadata is preserved
    for chunk in chunks:
        assert chunk.metadata["source"] == "test.txt"


def test_format_documents(sample_documents):
    """Test formatting documents into a context string."""
    context = format_documents(sample_documents)

    # Check all document content is included
    for doc in sample_documents:
        assert doc.page_content in context

    # Check documents are separated by newlines
    assert "\n\n" in context
