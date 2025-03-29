# Local RAG

A simple tool for Retrieval-Augmented Generation (RAG) using documents from your local filesystem.

## Overview

Local RAG allows you to:
1. Create vector embeddings from your local documents (PDF, TXT)
2. Query these documents using natural language, leveraging OpenAI's language models

## Requirements

- Python 3.13 or higher
- OpenAI API key (set in your .env file)

## Installation

```bash
# Clone the repository
git clone https://github.com/sualeh/local-rag.git
cd local-rag

# Install dependencies
pip install -e .
```

## Configuration

Create a `.env` file in the project root with your OpenAI API key:

```
OPENAI_API_KEY=your-api-key-here
```

You can also create a `.env.params` file to set default directories:

```
DOCS_DIRECTORY=path/to/your/documents
VECTOR_DB_PATH=path/to/save/vector/database
```

## Run Locally

Run locally with the following command, with the approproate arguments:

```bash
poetry run python -m local_rag.main
```


## Usage

### Create Vector Database

```bash
# Using command line arguments
python -m local_rag.main embed --docs-directory /path/to/docs --vector-db-path /path/to/vector_db

# Or if installed as a package
local-rag embed --docs-directory /path/to/docs --vector-db-path /path/to/vector_db
```

### Query Documents

```bash
# Using command line arguments
python -m local_rag.main query --vector-db-path /path/to/vector_db

# Or if installed as a package
local-rag query --vector-db-path /path/to/vector_db
```
