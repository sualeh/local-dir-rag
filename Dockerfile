FROM python:3.13-slim

WORKDIR /app

ENV \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY . .

RUN \
    pip install --upgrade pip && \
    pip install poetry

RUN \
    poetry config virtualenvs.create false && \
    poetry install --only main

# Create directories for mounting volumes
RUN \
    mkdir -p /data/docs /data/vector_db && \
    # Create a non-root user to run the application
    adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /app /data

ENV \
    DOCS_DIRECTORY=/data/docs \
    VECTOR_DB_PATH=/data/vector_db

# Switch to non-root user
USER appuser

ENTRYPOINT ["python", "-m", "local_rag.main"]
