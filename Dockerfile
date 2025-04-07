FROM python:3.13-slim

WORKDIR /app

# Set pip to not cache and use no color output
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copy only dependency files first to leverage build cache
COPY pyproject.toml ./

# Install system dependencies, Poetry, and project dependencies in one layer
RUN \
    pip install --upgrade pip && \
    pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev && \
    # Clean up cache to reduce image size
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy application code
COPY ./local_rag/ ./local_rag/

# Create directories for mounting volumes
RUN \
    mkdir -p /data/docs /data/vector_db && \
    # Create a non-root user to run the application
    adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /app /data

# Set default environment variables
ENV \
    DOCS_DIRECTORY=/data/docs \
    VECTOR_DB_PATH=/data/vector_db

# Switch to non-root user
USER appuser

# Set the entrypoint to run the CLI
ENTRYPOINT ["python", "-m", "local_rag.main"]
