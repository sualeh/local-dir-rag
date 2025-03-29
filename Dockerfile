FROM python:3.13-slim

WORKDIR /app

# Copy project files
COPY . .

# Install the package and dependencies
RUN pip install -e .

# Create directories for mounting volumes
RUN mkdir -p /data/docs /data/vector_db

# Set default environment variables
ENV DOCS_DIRECTORY=/data/docs
ENV VECTOR_DB_PATH=/data/vector_db

# Set the entrypoint to run the CLI
ENTRYPOINT ["python", "-m", "local_rag.main"]
