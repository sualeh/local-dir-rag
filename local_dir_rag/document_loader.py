"""Document Loader Module to load documents from directories."""
import glob

import os
import logging
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.schema import Document

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(None)


def get_files_from_directory(
    directory_path: str, extensions: list[str] = None
) -> list[str]:
    """
    Get all files with specified extensions from a directory.

    Args:
        directory_path: Path to the directory containing files.
        extensions: List of file extensions to include.

    Returns:
        List of absolute file paths matching the specified extensions.
    """

    if extensions is None:
        extensions = [".pdf", ".txt"]

    all_files = []
    for ext in extensions:
        files = glob.glob(os.path.join(directory_path, f"*{ext}"))
        all_files.extend(files)

    logger.info(
        "Found %d %s files in '%s'",
        len(all_files),
        str(extensions),
        directory_path
    )
    return all_files


def load_document(file_path: str) -> list[Document]:
    """
    Load a document based on its file extension.

    Args:
        file_path: Path to the file to be loaded.

    Returns:
        List of Document objects containing the content and metadata.
    """
    _, file_name = os.path.split(file_path)
    logger.info("Loading '%s'", file_name)
    if not os.path.isfile(file_path):
        logger.error("File not found: %s", file_path)
        return []
    _, file_extension = os.path.splitext(file_path)

    if file_extension.lower() == ".pdf":
        loader = PyPDFLoader(file_path)
        return loader.load()

    if file_extension.lower() == ".txt":
        loader = TextLoader(file_path)
        return loader.load()

    logger.error("Unsupported file format: %s", file_extension)
    return []


def print_document_chunks(documents: list[Document], limit: int = 3) -> None:
    """
    Print preview of document chunks with their metadata.

    Args:
        documents: List of Document objects to preview.
        limit: Maximum number of chunks to display.
    """
    print()
    for index, chunk in enumerate(documents):
        if index > limit:
            break
        print(
            f"------ CHUNK {index+1} ----------------------------------------"
        )
        print(chunk.metadata)
        print()
        print(chunk.page_content[:100])
        print("... (skipping content) ...")
        print(chunk.page_content[-100:])
        print()
