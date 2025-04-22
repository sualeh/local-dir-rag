import os
import pytest
from unittest.mock import patch, MagicMock
from langchain_community.vectorstores import FAISS
from local_dir_rag.query_with_rag import query_loop


def test_query_loop_loads_database(mock_faiss_db):
    """Test that query_loop loads the vector database."""
    with patch(
        "local_dir_rag.query_with_rag.load_vector_database"
    ) as mock_load:
        # Mock the interactive input to exit immediately
        with patch("builtins.input", return_value="exit"):
            query_loop(vector_db_path=mock_faiss_db)

        mock_load.assert_called_once_with(mock_faiss_db)


def test_query_processing(mock_faiss_db, mock_openai_embeddings):
    """Test the query processing workflow with mocked components."""
    # Mock the chat model response
    mock_chat = MagicMock()
    mock_chat.invoke.return_value = MagicMock()
    mock_chat.invoke.return_value.content = "This is a test response."

    with patch(
        "local_dir_rag.query_with_rag.load_vector_database"
    ) as mock_load:
        # Return a real vector db
        db = FAISS.load_local(
            mock_faiss_db,
            mock_openai_embeddings,
            allow_dangerous_deserialization=True
        )
        mock_load.return_value = db

        # Mock the chat model creation
        with patch(
            "local_dir_rag.query_with_rag.ChatOpenAI", return_value=mock_chat
        ):
            # Simulate a query and then exit
            inputs = ["What is artificial intelligence?", "exit"]

            # Mock the interactive input
            with patch("builtins.input", side_effect=inputs):
                # Call the query loop
                query_loop(vector_db_path=mock_faiss_db)

            # Check that the chat model was invoked
            assert mock_chat.invoke.called
