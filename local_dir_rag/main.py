"""Main entry point for the local-dir-rag package."""

import argparse
import os
import logging
from dotenv import load_dotenv
from local_dir_rag.query_with_rag import query_loop
from local_dir_rag.embed import embed_docs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s `%(funcName)s` %(levelname)s:\n  %(message)s"
)
logger = logging.getLogger(__name__)


def embed(docs_paths: str | list[str] = None, vector_db_path: str = None):
    """
    Create and save a vector database from documents.

    Args:
        docs_paths (str | list[str], optional): One or more document
            directories. Strings may contain multiple paths separated by
            ``os.pathsep``.
        vector_db_path (str, optional): Path to save the vector database.
    """
    docs_paths = docs_paths or os.getenv("DOCS_PATH")
    if docs_paths is None:
        raise ValueError("Documents path is not set.")

    vector_db_path = vector_db_path or os.getenv("VECTOR_DB_PATH")
    if vector_db_path is None:
        raise ValueError("Vector database path is not set.")

    return embed_docs(
        docs_paths=docs_paths,
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
        "--docs-paths",
        "--docs-directory",
        dest="docs_paths",
        required=False,
        help=(
            "One or more directories containing documents to embed. "
            f"Separate multiple paths with '{os.pathsep}'."
        )
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
        embed(args.docs_paths, args.vector_db_path)
    elif args.command == "query":
        query(args.vector_db_path)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
