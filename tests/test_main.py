import os
import pytest
from unittest.mock import patch
from local_dir_rag.main import main, embed, query


def test_embed_function(temp_dir):
    """Test the embed function."""
    docs_dir = os.path.join(temp_dir, "docs")
    os.makedirs(docs_dir, exist_ok=True)
    vector_db_path = os.path.join(temp_dir, "vector_db")

    # Create a test file
    with open(os.path.join(docs_dir, "test.txt"), "w", encoding="utf-8") as f:
        f.write("Test document content")

    # Mock the embed_docs function to avoid actual embedding
    with patch("local_dir_rag.main.embed_docs") as mock_embed:
        embed(docs_directory=docs_dir, vector_db_path=vector_db_path)
        mock_embed.assert_called_once_with(
            docs_directory=docs_dir,
            vector_db_path=vector_db_path
        )


def test_query_function(temp_dir):
    """Test the query function."""
    vector_db_path = os.path.join(temp_dir, "vector_db")
    os.makedirs(vector_db_path, exist_ok=True)

    # Mock the query_loop function to avoid actual querying
    with patch("local_dir_rag.main.query_loop") as mock_query:
        query(vector_db_path=vector_db_path)
        mock_query.assert_called_once_with(vector_db_path=vector_db_path)


def test_main_embed_command():
    """Test main function with embed command."""
    test_args = [
        "embed",
        "--docs-directory", "/test/docs",
        "--vector-db-path", "/test/vector_db"
    ]

    with patch("sys.argv", ["local_dir_rag"] + test_args):
        with patch("local_dir_rag.main.embed") as mock_embed:
            main()
            mock_embed.assert_called_once_with(
                docs_directory="/test/docs",
                vector_db_path="/test/vector_db"
            )


def test_main_query_command():
    """Test main function with query command."""
    test_args = ["query", "--vector-db-path", "/test/vector_db"]

    with patch("sys.argv", ["local_dir_rag"] + test_args):
        with patch("local_dir_rag.main.query") as mock_query:
            main()
            mock_query.assert_called_once_with(vector_db_path="/test/vector_db")


def test_main_no_command(capfd):
    """Test main function with no command."""
    with patch("sys.argv", ["local_dir_rag"]):
        with patch(
            "local_dir_rag.main.argparse.ArgumentParser.print_help"
        ) as mock_help:
            main()
            mock_help.assert_called_once()
