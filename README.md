# Local Directory RAG

A simple tool for Retrieval-Augmented Generation (RAG) using documents from your local filesystem.

## Overview

Local Directory RAG allows you to:

1. Create vector embeddings from your local documents (PDF, TXT)
2. Query these documents using natural language, leveraging OpenAI's language models

## Requirements

- Python 3.13 or higher
- OpenAI API key (set in your .env file)

## Installation

```bash
# Clone the repository
git clone https://github.com/sualeh/local-dir-rag.git
cd local-dir-rag

# Install dependencies
pip install -e .
```

## Configuration

Create a `.env` file in the project root with your OpenAI API key:

```bash
OPENAI_API_KEY=your-api-key-here
```

You can also create a `.env.params` file to set default directories:

```bash
DOCS_DIRECTORY=path/to/your/documents
VECTOR_DB_PATH=path/to/save/vector/database
```

## Run Locally

Run locally with the following command, with the approproate arguments:

```bash
poetry run python -m local_dir_rag.main
```

## Usage

### Create Vector Database

```bash
# Using command line arguments
python -m local_dir_rag.main embed --docs-directory /path/to/docs --vector-db-path /path/to/vector_db

# Or if installed as a package
local-dir-rag embed --docs-directory /path/to/docs --vector-db-path /path/to/vector_db
```

### Query Documents

```bash
# Using command line arguments
python -m local_dir_rag.main query --vector-db-path /path/to/vector_db

# Or if installed as a package
local-dir-rag query --vector-db-path /path/to/vector_db
```

## Docker Usage

You can run Local RAG using Docker:

```bash
# Pull the Docker image
docker pull sualeh/local-dir-rag:latest

# Embed documents
docker run -v /path/to/your/docs:/data/docs -v /path/to/vector_db:/data/vector_db \
  -e OPENAI_API_KEY=your-api-key-here \
  sualeh/local-dir-rag embed

# Query your documents
docker run -v /path/to/vector_db:/data/vector_db \
  -e OPENAI_API_KEY=your-api-key-here \
  sualeh/local-dir-rag query
```

You can also pass command line arguments directly:

```bash
docker run -v /path/to/your/docs:/data/docs -v /path/to/output:/data/vector_db \
  -e OPENAI_API_KEY=your-api-key-here \
  sualeh/local-dir-rag embed --docs-directory /data/docs --vector-db-path /data/vector_db
```

## Docker Compose Usage

You can also use Docker Compose for easier management of the Local RAG container:

1. Create a `docker-compose.yml` file:

```yaml
version: '3'
services:
  local-dir-rag:
    image: sualeh/local-dir-rag:latest
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./docs:/data/docs
      - ./vector_db:/data/vector_db
```

2. Run the application:

```bash
# For embedding documents
docker-compose run local-dir-rag embed

# For querying documents
docker-compose run local-dir-rag query
```

You can also pass additional arguments:

```bash
docker-compose run local-dir-rag embed --docs-directory /data/docs --vector-db-path /data/vector_db
```

This approach simplifies volume mounting and environment variable management, especially when working with the tool regularly.
