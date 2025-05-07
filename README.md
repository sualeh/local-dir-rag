# Local Directory RAG

A simple tool for Retrieval-Augmented Generation (RAG) using documents from your local filesystem.

## Overview

Local Directory RAG allows you to:

1. Create vector embeddings from your local documents (PDF, TXT)
2. Query these documents using natural language, leveraging OpenAI's language models

## Requirements

- Python 3.13 or higher
- OpenAI API key and other parameters (set in your .env file)

## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management.

1. Install Poetry by following the instructions in the [official documentation](https://python-poetry.org/docs/#installation).

    Quick installation methods:

    ```bash
    # For Linux, macOS, Windows (WSL)
    curl -sSL https://install.python-poetry.org | python3 -
    ```

    ```pwsh
    # For Windows PowerShell
    (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
    ```

2. Install Project Dependencies

    ```bash
    # Clone the repository
    git clone https://github.com/sualeh/local-dir-rag.git
    cd local-dir-rag
    ```

3. Install dependencies using Poetry

    ```bash
    poetry install --extras "dev"
    poetry show --tree
    ```


## Configuration

Copy the ".env.example" file as ".env" in the project root. Update it with your OpenAI API key, location of your documents, and where you would like the vector database to be created.


## Usage

1. Create Vector Database

    ```bash
    poetry run python -m local_dir_rag.main embed --docs-directory /path/to/docs --vector-db-path /path/to/vector_db
    ```

2. Query Documents

    ```bash
    poetry run python -m local_dir_rag.main query --vector-db-path /path/to/vector_db
    ```


## Development and Testing

1. Install dependencies, as above.

2. Run all tests:

    ```bash
    poetry run pytest
    ```

    Or, run a single test:

    ```bash
    poetry run pytest tests/test_document_loader.py::test_load_document
    ```


## Docker Compose Usage

You can also use Docker Compose for easier management of the Local RAG container:

1. Clone the project, as described above.

2. Configure the ".env" file as described above.

3. Run the application using Docker Compose:

      ```bash
      # For embedding documents
      docker-compose run local-dir-rag embed
      ```

      ```bash
      # For querying documents
      docker-compose run local-dir-rag query
      ```

      You can also pass additional arguments:

      ```bash
      docker-compose run local-dir-rag embed --docs-directory /data/docs --vector-db-path /data/vector_db
      ```

This approach simplifies volume mounting and environment variable management, especially when working with the tool regularly.
