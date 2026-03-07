import pytest
from pathlib import Path

from src.ingestion.hash_tracker import IngestionTracker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tracker(tmp_path):
    """Create a fresh IngestionTracker with a throwaway SQLite DB."""
    db_path = str(tmp_path / "test_ingestion_state.db")
    t = IngestionTracker(db_path)
    yield t
    t.close()


@pytest.fixture
def sample_file(tmp_path):
    """Create a small temp file with known content."""
    f = tmp_path / "sample_deal.pdf"
    f.write_bytes(b"CREDIT AGREEMENT between Amerigo Resources Ltd. and RBC")
    return f


# ===========================================================================
# Test 1 — Database Creation
# ===========================================================================

def test_db_creation(tmp_path):
    """The SQLite database file must exist after tracker initialization."""
    db_path = tmp_path / "state.db"
    t = IngestionTracker(str(db_path))
    assert db_path.exists(), "SQLite DB file was not created"
    t.close()


# ===========================================================================
# Test 2 — Hash Determinism
# ===========================================================================

def test_compute_hash_deterministic(sample_file):
    """Hashing the same file twice must produce identical digests."""
    hash_1 = IngestionTracker.compute_hash(str(sample_file))
    hash_2 = IngestionTracker.compute_hash(str(sample_file))
    assert hash_1 == hash_2


# ===========================================================================
# Test 3 — Hash Sensitivity to Content
# ===========================================================================

def test_compute_hash_changes_with_content(tmp_path):
    """Two files with different content must produce different hashes."""
    file_a = tmp_path / "deal_a.pdf"
    file_b = tmp_path / "deal_b.pdf"
    file_a.write_bytes(b"Term Sheet Version 1")
    file_b.write_bytes(b"Term Sheet Version 2")

    assert IngestionTracker.compute_hash(str(file_a)) != IngestionTracker.compute_hash(str(file_b))


# ===========================================================================
# Test 4 — New File is Not Already Processed
# ===========================================================================

def test_new_file_is_not_already_processed(tracker, sample_file):
    """A file that has never been tracked must return False."""
    assert tracker.is_already_processed(str(sample_file)) is False


# ===========================================================================
# Test 5 — Mark Then Check Returns True
# ===========================================================================

def test_mark_then_check_returns_true(tracker, sample_file):
    """After marking a file as processed, is_already_processed must return True."""
    tracker.mark_as_processed(str(sample_file))
    assert tracker.is_already_processed(str(sample_file)) is True


# ===========================================================================
# Test 6 — Modified File is Detected
# ===========================================================================

def test_modified_file_is_detected(tracker, sample_file):
    """
    If a file is marked as processed and then its content changes,
    is_already_processed must return False (hash mismatch).
    """
    tracker.mark_as_processed(str(sample_file))

    # Overwrite the file with different content (simulates a user replacing the PDF)
    sample_file.write_bytes(b"AMENDED CREDIT AGREEMENT - new terms added")

    assert tracker.is_already_processed(str(sample_file)) is False

# ===========================================================================
# Test 7 — Deletion Logic
# ===========================================================================

def test_remove_from_tracker(tracker, sample_file):
    """Verify that removing a file from the tracker works."""
    tracker.mark_as_processed(str(sample_file))
    assert tracker.is_already_processed(str(sample_file)) is True
    
    filename = sample_file.name
    success = tracker.remove_from_tracker(filename)
    
    assert success is True
    assert tracker.is_already_processed(str(sample_file)) is False

# ===========================================================================
# Test 8 — Persistence Across Sessions
# ===========================================================================

def test_close_and_reopen_persists(tmp_path, sample_file):
    """
    Data must survive closing and reopening the tracker.
    This proves SQLite commits are durable, not just in-memory.
    """
    db_path = str(tmp_path / "persist_test.db")

    # Session 1: mark the file
    tracker_1 = IngestionTracker(db_path)
    tracker_1.mark_as_processed(str(sample_file))
    tracker_1.close()

    # Session 2: new tracker instance, same DB
    tracker_2 = IngestionTracker(db_path)
    assert tracker_2.is_already_processed(str(sample_file)) is True
    tracker_2.close()
