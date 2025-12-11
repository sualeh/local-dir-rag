# AI Agent Guide

Project-specific guidance to make AI contributions consistent and productive.

## Architecture and Flow

- CLI entrypoint `local_dir_rag.main`: subcommands `embed` and `query`; invoked via `python -m local_dir_rag.main` or script `local-dir-rag`.
- Embedding pipeline (`embed_docs` in `local_dir_rag/embed.py`): load files from `DOCS_PATH` (supports multiple paths separated by `os.pathsep`), accept `.pdf`/`.txt` only, split into chunks (`text_processor.split_documents`), store in FAISS via `vector_store.load_vector_database` (append if existing) and `FAISS.save_local(vector_db_path)`.
- Query pipeline (`query_with_rag.py`): load FAISS from `VECTOR_DB_PATH`, retriever `k=10`, prompt template inside file, ChatOpenAI `model_name="gpt-4o"`, `temperature=0.7`; prints sources via `print_sources` (metadata keys stripped, filenames shortened) before formatting context.
- Vector store loader (`vector_store.load_vector_database`): uses `OpenAIEmbeddings`; `allow_dangerous_deserialization=True` required by FAISS load.

## Configuration and Secrets

- Required env vars: `OPENAI_API_KEY`, `DOCS_PATH`, `VECTOR_DB_PATH`; sample in `.env.example`. `.env` is loaded in `main.py` and `query_with_rag.py` via `python-dotenv`.
- For Docker: container expects docs at `/data/docs`, vector DB at `/data/vector_db`; compose mounts `${DOCS_PATH}` and `${VECTOR_DB_PATH}` into those paths (use a single path when running via Docker Compose).

## Workflows

- Install (Poetry): `poetry install --extras "dev"`; app deps only: `poetry install --only main` (used in Dockerfile).
- Embed: `poetry run python -m local_dir_rag.main embed --docs-directory /path/to/docs --vector-db-path /path/to/vector_db` (flags optional if env vars set).
- Query: `poetry run python -m local_dir_rag.main query --vector-db-path /path/to/vector_db`; interactive loop until `exit`/`quit`.
- Tests: `poetry run pytest` (coverage enabled); focused example `poetry run pytest tests/test_document_loader.py::test_load_document`.
- Docker: `docker-compose run local-dir-rag embed ...` or `query ...`; image `sualehfatehi/local-dir-rag:latest`.

## Conventions

- Style: PEP 8, 4-space indent, max line length 80, trim trailing whitespace, newline at EOF.
- Docstrings: Google style for functions/classes.
- Logging: use placeholder logging (no f-strings); `logging.basicConfig` already set in modules—avoid redefining.
- String formatting: prefer f-strings only for non-logging simple cases.
- Types: add type hints where reasonable (e.g., `list[Document]`, `Embeddings`).
- Chunking defaults: `split_documents` uses `RecursiveCharacterTextSplitter` with chunk size 1024 and overlap 150; preserve these unless a clear reason to change.
- Supported loaders: `.pdf` via `PyPDFLoader`, `.txt` via `TextLoader`; unsupported formats should log error and return empty list (see tests).
- Metadata hygiene: before printing sources, drop `producer/creator/creationdate/moddate/total_pages`; shorten `source` to basename.

## Testing Patterns

- Tests live in `tests/`; fixtures in `tests/conftest.py` create temp docs. Respect expectations (e.g., `.pdf` not present in fixtures; `.txt` count = 3).
- Keep new tests fast and isolated; reuse fixture shapes (Document metadata includes `source`, `page`).

## External Dependencies

- LangChain stack: `langchain`, `langchain-community`, `langchain-openai`, `langchain-text-splitters`, `faiss-cpu`, `sentence-transformers`.
- OpenAI models: embeddings via `OpenAIEmbeddings`, chat via `ChatOpenAI`; ensure keys pulled from env.

## When Extending

- Add new file types by extending `document_loader.load_document` and updating tests.
- If changing prompt or model params, keep safety: answer only from context when unknown.
- When altering vector store behavior, preserve incremental save/append semantics and backward compatibility with existing FAISS directories.
