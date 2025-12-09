"""Tests for the file tracker module."""
import os
import sqlite3

from local_dir_rag.file_tracker import (
    FileTracker,
    FileStatus,
    compute_file_checksum,
)


def test_compute_file_checksum(temp_dir):
    """Test that checksum is computed correctly."""
    file_path = os.path.join(temp_dir, "test.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("test content")

    checksum1 = compute_file_checksum(file_path)
    assert isinstance(checksum1, str)
    assert len(checksum1) == 64  # SHA-256 hex digest

    # Same content should produce same checksum
    checksum2 = compute_file_checksum(file_path)
    assert checksum1 == checksum2

    # Different content should produce different checksum
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("different content")

    checksum3 = compute_file_checksum(file_path)
    assert checksum1 != checksum3


def test_file_tracker_init(temp_dir):
    """Test that file tracker initializes database correctly."""
    vector_db_path = os.path.join(temp_dir, "vector_db")
    tracker = FileTracker(vector_db_path)

    # Directory should be created
    assert os.path.exists(vector_db_path)

    # Database file should exist
    assert os.path.exists(tracker.db_path)
    assert tracker.db_path == os.path.join(vector_db_path, "file_tracker.db")

    # Table should exist
    with sqlite3.connect(tracker.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='file_checksums'"
        )
        assert cursor.fetchone() is not None


def test_file_status_new_file(temp_dir):
    """Test that new files are correctly identified."""
    vector_db_path = os.path.join(temp_dir, "vector_db")
    tracker = FileTracker(vector_db_path)

    # Create a test file
    file_path = os.path.join(temp_dir, "new_file.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("new file content")

    status = tracker.get_file_status(file_path)
    assert status.is_new is True
    assert status.is_modified is False
    assert status.needs_indexing is True


def test_file_status_unchanged_file(temp_dir):
    """Test that unchanged files are correctly identified."""
    vector_db_path = os.path.join(temp_dir, "vector_db")
    tracker = FileTracker(vector_db_path)

    # Create and track a file
    file_path = os.path.join(temp_dir, "tracked_file.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("tracked file content")

    tracker.update_file_checksum(file_path)

    # Check status without modifying
    status = tracker.get_file_status(file_path)
    assert status.is_new is False
    assert status.is_modified is False
    assert status.needs_indexing is False


def test_file_status_modified_file(temp_dir):
    """Test that modified files are correctly identified."""
    vector_db_path = os.path.join(temp_dir, "vector_db")
    tracker = FileTracker(vector_db_path)

    # Create and track a file
    file_path = os.path.join(temp_dir, "modified_file.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("original content")

    tracker.update_file_checksum(file_path)

    # Modify the file
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("modified content")

    status = tracker.get_file_status(file_path)
    assert status.is_new is False
    assert status.is_modified is True
    assert status.needs_indexing is True


def test_update_file_checksum(temp_dir):
    """Test updating file checksum."""
    vector_db_path = os.path.join(temp_dir, "vector_db")
    tracker = FileTracker(vector_db_path)

    file_path = os.path.join(temp_dir, "update_test.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("initial content")

    # First update
    tracker.update_file_checksum(file_path)
    status1 = tracker.get_file_status(file_path)
    assert status1.is_new is False
    assert status1.is_modified is False

    # Modify and update again
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("updated content")

    # Before update, should be modified
    status2 = tracker.get_file_status(file_path)
    assert status2.is_modified is True

    # After update, should not be modified
    tracker.update_file_checksum(file_path)
    status3 = tracker.get_file_status(file_path)
    assert status3.is_modified is False


def test_remove_file(temp_dir):
    """Test removing a file from the tracker."""
    vector_db_path = os.path.join(temp_dir, "vector_db")
    tracker = FileTracker(vector_db_path)

    file_path = os.path.join(temp_dir, "to_remove.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("content to remove")

    tracker.update_file_checksum(file_path)
    assert file_path in tracker.get_all_tracked_files()

    tracker.remove_file(file_path)
    assert file_path not in tracker.get_all_tracked_files()

    # File should be seen as new again
    status = tracker.get_file_status(file_path)
    assert status.is_new is True


def test_get_all_tracked_files(temp_dir):
    """Test getting all tracked files."""
    vector_db_path = os.path.join(temp_dir, "vector_db")
    tracker = FileTracker(vector_db_path)

    # Initially empty
    assert tracker.get_all_tracked_files() == []

    # Add some files
    file_paths = []
    for i in range(3):
        file_path = os.path.join(temp_dir, f"file_{i}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"content {i}")
        tracker.update_file_checksum(file_path)
        file_paths.append(file_path)

    tracked = tracker.get_all_tracked_files()
    assert len(tracked) == 3
    for fp in file_paths:
        assert fp in tracked


def test_get_deleted_files(temp_dir):
    """Test finding deleted files."""
    vector_db_path = os.path.join(temp_dir, "vector_db")
    tracker = FileTracker(vector_db_path)

    # Track some files
    file1 = os.path.join(temp_dir, "file1.txt")
    file2 = os.path.join(temp_dir, "file2.txt")
    file3 = os.path.join(temp_dir, "file3.txt")

    for fp in [file1, file2, file3]:
        with open(fp, "w", encoding="utf-8") as f:
            f.write("content")
        tracker.update_file_checksum(fp)

    # Simulate file2 being deleted (not in current files list)
    current_files = [file1, file3]
    deleted = tracker.get_deleted_files(current_files)

    assert len(deleted) == 1
    assert file2 in deleted


def test_file_status_dataclass():
    """Test FileStatus dataclass properties."""
    # New file needs indexing
    status_new = FileStatus(
        file_path="/path/to/file.txt",
        is_new=True,
        is_modified=False
    )
    assert status_new.needs_indexing is True

    # Modified file needs indexing
    status_modified = FileStatus(
        file_path="/path/to/file.txt",
        is_new=False,
        is_modified=True
    )
    assert status_modified.needs_indexing is True

    # Unchanged file does not need indexing
    status_unchanged = FileStatus(
        file_path="/path/to/file.txt",
        is_new=False,
        is_modified=False
    )
    assert status_unchanged.needs_indexing is False
