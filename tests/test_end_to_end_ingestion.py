import pytest
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.rag.chroma_deal_store import initialize_vector_store
from langchain_core.documents import Document

@patch("src.rag.chroma_deal_store.Chroma")
@patch("src.rag.chroma_deal_store.OllamaEmbeddings")
@patch("src.rag.chroma_deal_store.RecursiveCharacterTextSplitter")
def test_full_ingestion_lifecycle(mock_splitter_class, mock_embeddings, mock_chroma, tmp_path):
    """
    Integration Test:
    1. Ingest a new file.
    2. Verify it's marked as processed in SQLite.
    3. Verify access_group='general' is injected.
    4. Run again and verify it's skipped.
    """
    # 1. Setup Paths
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    test_file = processed_dir / "deal_1.txt"
    test_file.write_text("Deal content")
    
    chroma_dir = tmp_path / "chroma"
    chroma_dir.mkdir()
    hash_db = tmp_path / "ingestion_tracker.db"

    # 2. Setup Mocks
    mock_vs = MagicMock()
    mock_chroma.return_value = mock_vs
    
    mock_splitter = MagicMock()
    mock_splitter_class.return_value = mock_splitter
    mock_splitter.split_documents.side_effect = lambda docs: docs

    # Patch module-level constants in chroma_deal_store.py
    with patch("src.rag.chroma_deal_store.PROCESSED_DATA_DIR", processed_dir), \
         patch("src.rag.chroma_deal_store.CHROMA_DB_DIR", chroma_dir), \
         patch("src.rag.chroma_deal_store.HASH_DB_PATH", hash_db), \
         patch("src.rag.chroma_deal_store.PROJECT_ROOT", tmp_path):
        
        # --- RUN 1: New File ---
        initialize_vector_store()
        
        # Verify Chroma received the document
        assert mock_vs.add_documents.called
        args, kwargs = mock_vs.add_documents.call_args
        ingested_docs = kwargs.get("documents", args[0] if args else [])
        assert ingested_docs[0].metadata["access_group"] == "general"
        assert ingested_docs[0].metadata["source"] == "deal_1.pdf"
        
        # Verify tracker recorded it
        from src.ingestion.hash_tracker import IngestionTracker
        tracker = IngestionTracker(str(hash_db))
        assert tracker.is_already_processed(str(test_file)) is True
        tracker.close()

        # --- RUN 2: Duplicate File (Incremental Sync) ---
        mock_vs.add_documents.reset_mock()
        initialize_vector_store()
        
        # Should NOT be called because hash hasn't changed
        assert not mock_vs.add_documents.called
