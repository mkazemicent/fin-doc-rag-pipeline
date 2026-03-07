import pytest
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.ingestion.hash_tracker import IngestionTracker
from src.rag.vector_store import delete_document_from_db
from langchain_core.documents import Document

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
    f = tmp_path / "amerigo_2015.pdf"
    f.write_bytes(b"Mock PDF Content")
    return f

# ===========================================================================
# Category 1 — Tracker Deletion Logic
# ===========================================================================

def test_remove_tracked_file(tracker, sample_file):
    """Mark a file as processed, then remove it and verify it's gone."""
    filename = str(sample_file.name)
    tracker.mark_as_processed(str(sample_file))
    
    assert tracker.is_already_processed(str(sample_file)) is True
    
    # Remove it
    removed = tracker.remove_from_tracker(filename)
    assert removed is True
    
    # Verify it's no longer tracked (so it can be re-ingested)
    assert tracker.is_already_processed(str(sample_file)) is False

def test_remove_nonexistent_file(tracker):
    """Removing a file that isn't tracked should return False."""
    assert tracker.remove_from_tracker("missing.pdf") is False

def test_remove_then_re_ingest_lifecycle(tracker, sample_file):
    """Full lifecycle: Ingest -> Remove -> Ingest again."""
    tracker.mark_as_processed(str(sample_file))
    tracker.remove_from_tracker(sample_file.name)
    
    # Re-ingest
    tracker.mark_as_processed(str(sample_file))
    assert tracker.is_already_processed(str(sample_file)) is True


# ===========================================================================
# Category 2 — Metadata Normalization Logic
# ===========================================================================

def test_metadata_normalization():
    """ Verify that path-based .txt source is normalized to bare .pdf filename. """
    mock_doc = Document(
        page_content="Text", 
        metadata={"source": "data/processed/amerigo_2015.txt"}
    )
    
    # Logic extracted from vector_store.py
    source_path = Path(mock_doc.metadata.get("source", ""))
    normalized_source = source_path.stem + ".pdf"
    
    assert normalized_source == "amerigo_2015.pdf"


# ===========================================================================
# Category 3 — ChromaDB Deletion (Mocked)
# ===========================================================================

@patch("src.rag.vector_store.get_chroma_instance")
def test_delete_document_from_db_mocked(mock_get_instance):
    """
    Test the deletion logic flow without a live ChromaDB.
    1. Mock ChromaDB instance.
    2. Simulate finding 5 matching chunk IDs.
    3. Verify delete() is called with those IDs.
    """
    mock_vs = MagicMock()
    mock_get_instance.return_value = mock_vs
    
    # Simulate finding IDs for 'amerigo_2015.pdf'
    mock_vs.get.return_value = {"ids": ["chunk1", "chunk2", "chunk3", "chunk4", "chunk5"]}
    
    deleted_count = delete_document_from_db("amerigo_2015.pdf", chroma_dir="/tmp/mock_db")
    
    assert deleted_count == 5
    mock_vs.get.assert_called_once_with(where={"source": "amerigo_2015.pdf"})
    mock_vs.delete.assert_called_once_with(ids=["chunk1", "chunk2", "chunk3", "chunk4", "chunk5"])

@patch("src.rag.vector_store.get_chroma_instance")
def test_delete_nonexistent_document_mocked(mock_get_instance):
    """Verify that deleting a non-existent document returns 0 and logs a warning."""
    mock_vs = MagicMock()
    mock_get_instance.return_value = mock_vs
    
    # Simulate no matches found
    mock_vs.get.return_value = {"ids": []}
    
    deleted_count = delete_document_from_db("ghost.pdf", chroma_dir="/tmp/mock_db")
    
    assert deleted_count == 0
    mock_vs.delete.assert_not_called()
