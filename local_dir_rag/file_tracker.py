"""File tracking module using SQLite for incremental indexing."""

import hashlib
import logging
import os
import sqlite3
from dataclasses import dataclass
from enum import Enum

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s `%(funcName)s` %(levelname)s:\n  %(message)s"
)
logger = logging.getLogger(__name__)


class FileState(Enum):
    """Enumeration of possible file states relative to the tracker."""
    NEW = "new"
    MODIFIED = "modified"
    UNCHANGED = "unchanged"


@dataclass
class FileStatus:
    """Status of a file relative to the tracker database."""
    file_path: str
    state: FileState

    @property
    def is_new(self) -> bool:
        """Check if the file is new."""
        return self.state == FileState.NEW

    @property
    def is_modified(self) -> bool:
        """Check if the file has been modified."""
        return self.state == FileState.MODIFIED

    @property
    def needs_indexing(self) -> bool:
        """Check if the file needs to be indexed."""
        return self.state in (FileState.NEW, FileState.MODIFIED)


def compute_file_checksum(file_path: str) -> str:
    """
    Compute SHA-256 checksum of a file.

    Args:
        file_path: Path to the file.

    Returns:
        SHA-256 hex digest of the file content.
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


class FileTracker:
    """
    Track file checksums in a SQLite database for incremental indexing.

    The database is stored alongside the vector store in the same directory.
    """

    def __init__(self, vector_db_path: str):
        """
        Initialize the file tracker.

        Args:
            vector_db_path: Path to the vector database directory.
                The SQLite database will be created in this directory.
        """
        self.vector_db_path = vector_db_path
        self.db_path = os.path.join(vector_db_path, "file_tracker.db")
        self._ensure_directory()
        self._init_database()

    def _ensure_directory(self) -> None:
        """Ensure the vector database directory exists."""
        os.makedirs(self.vector_db_path, exist_ok=True)

    def _init_database(self) -> None:
        """Initialize the SQLite database with the required schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS file_checksums (
                    file_path TEXT PRIMARY KEY,
                    directory_path TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
        logger.info("File tracker database initialized at %s", self.db_path)

    def get_file_status(self, file_path: str) -> FileStatus:
        """
        Get the status of a file relative to what's stored in the database.

        Args:
            file_path: Absolute path to the file.

        Returns:
            FileStatus indicating if the file is new or modified.
        """
        current_checksum = compute_file_checksum(file_path)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT checksum FROM file_checksums WHERE file_path = ?",
                (file_path,)
            )
            row = cursor.fetchone()

        if row is None:
            return FileStatus(
                file_path=file_path,
                state=FileState.NEW
            )

        stored_checksum = row[0]
        if current_checksum != stored_checksum:
            return FileStatus(
                file_path=file_path,
                state=FileState.MODIFIED
            )

        return FileStatus(
            file_path=file_path,
            state=FileState.UNCHANGED
        )

    def update_file_checksum(self, file_path: str) -> None:
        """
        Update or insert the checksum for a file.

        Args:
            file_path: Absolute path to the file.
        """
        checksum = compute_file_checksum(file_path)
        directory_path, file_name = os.path.split(file_path)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO file_checksums
                    (file_path, directory_path, file_name, checksum)
                VALUES (?, ?, ?, ?)
            """, (file_path, directory_path, file_name, checksum))
            conn.commit()

        logger.info("Updated checksum for %s", file_path)

    def remove_file(self, file_path: str) -> None:
        """
        Remove a file from the tracker database.

        Args:
            file_path: Absolute path to the file.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM file_checksums WHERE file_path = ?",
                (file_path,)
            )
            conn.commit()

        logger.info("Removed %s from tracker", file_path)

    def get_all_tracked_files(self) -> list[str]:
        """
        Get all file paths currently tracked in the database.

        Returns:
            List of file paths.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT file_path FROM file_checksums")
            rows = cursor.fetchall()

        return [row[0] for row in rows]

    def get_deleted_files(self, current_files: list[str]) -> list[str]:
        """
        Find files that are tracked but no longer exist in current_files.

        Args:
            current_files: List of current file paths in the docs directory.

        Returns:
            List of file paths that were tracked but are now deleted.
        """
        tracked_files = set(self.get_all_tracked_files())
        current_files_set = set(current_files)
        return list(tracked_files - current_files_set)
