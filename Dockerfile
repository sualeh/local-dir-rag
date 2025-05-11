FROM python:3.13-slim

# Set the working directory inside the container to /app
# Create the directory if it doesn't exist
WORKDIR /app

# Set environment variables to configure pip and Python behavior
ENV \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install required Python package management tools
RUN \
    pip install --upgrade pip && \
    pip install poetry

# Copy application code from local directory to container
COPY . .

# Configure Poetry and install dependencies
RUN \
    poetry config virtualenvs.create false && \
    poetry install --only main

# Create directories for mounting volumes
# Create a non-root user and set permissions
RUN \
    mkdir -p /data/docs /data/vector_db && \
    adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /app /data

ENV \
    DOCS_DIRECTORY=/data/docs \
    VECTOR_DB_PATH=/data/vector_db

# Switch to non-root user
USER appuser

ENTRYPOINT ["python", "-m", "local_dir_rag.main"]
