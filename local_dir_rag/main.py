"""Main entry point for the local-dir-rag package."""

import argparse
import os
import logging
from dotenv import load_dotenv
from local_dir_rag.query_with_rag import query_loop as query_loop
from local_dir_rag.embed import embed as embed_docs

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(None)


def embed(docs_directory: str = None, vector_db_path: str = None):
    """
    Create and save a vector database from documents.

    Args:
        docs_directory (str, optional): Directory containing documents to
            process.
        vector_db_path (str, optional): Path to save the vector database.
    """
    docs_directory = docs_directory or os.getenv("DOCS_DIRECTORY")
    if docs_directory is None:
        raise ValueError("Documents directory is not set.")

    vector_db_path = vector_db_path or os.getenv("VECTOR_DB_PATH")
    if vector_db_path is None:
        raise ValueError("Vector database path is not set.")

    return embed_docs(
        docs_directory=docs_directory,
        vector_db_path=vector_db_path
    )


def query(vector_db_path: str = None):
    """
    Run an interactive query session using the specified vector database.

    Args:
        vector_db_path: Path to the vector database to query
    """
    vector_db_path = vector_db_path or os.getenv("VECTOR_DB_PATH")
    if vector_db_path is None:
        raise ValueError("Vector database path is not set.")

    query_loop(vector_db_path)


def main():
    """
    Main entry point for the application.
    Parses command-line arguments and executes the appropriate command.
    """
    # Load environment variables
    load_dotenv()
    load_dotenv(dotenv_path=".env.params")

    # Set up the argument parser
    parser = argparse.ArgumentParser(
        description="Local RAG system for processing and querying documents"
    )
    subparsers = parser.add_subparsers(
        dest="command",
        help="Command to execute"
    )

    # Parser for the embed command
    embed_parser = subparsers.add_parser(
        "embed",
        help="Create a vector database from documents"
    )
    embed_parser.add_argument(
        "--docs-directory",
        required=False,
        help="Directory containing documents to embed"
    )
    embed_parser.add_argument(
        "--vector-db-path",
        required=False,
        help="Path where to save the vector database"
    )

    # Parser for the query command
    query_parser = subparsers.add_parser(
        "query",
        help="Query an existing vector database"
    )
    query_parser.add_argument(
        "--vector-db-path",
        required=False,
        help="Path to the vector database to query"
    )

    # Parse the arguments and execute the appropriate command
    args = parser.parse_args()
    if args.command == "embed":
        embed(args.docs_directory, args.vector_db_path)
    elif args.command == "query":
        query(args.vector_db_path)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
