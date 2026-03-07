import pytest
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.rag.vector_store import initialize_vector_store
from langchain_core.documents import Document

@pytest.fixture
def mock_env(tmp_path):
    """Setup mock environment and directories."""
    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)
    
    chroma_dir = tmp_path / "data" / "chroma_db"
    chroma_dir.mkdir(parents=True)
    
    hash_db = tmp_path / "data" / "ingestion_tracker.db"
    
    # Dummy file to ingest
    test_file = processed_dir / "deal_1.txt"
    test_file.write_text("This is a deal document content.")
    
    with patch.dict(os.environ, {
        "EMBEDDING_MODEL": "mxbai-embed-large",
        "OLLAMA_BASE_URL": "http://localhost:11434"
    }):
        with patch("src.rag.vector_store.Path") as mock_path:
            # Configure mock paths to point to our temp directory
            def side_effect(path_str):
                if "data/processed" in str(path_str): return processed_dir
                if "data/chroma_db" in str(path_str): return chroma_dir
                if "data/ingestion_tracker.db" in str(path_str): return hash_db
                return Path(path_str)
            
            # We need to be careful with Path mocking. 
            # Instead of mocking Path entirely, let's mock the constants in vector_store
            yield {
                "processed_dir": processed_dir,
                "chroma_dir": chroma_dir,
                "hash_db": hash_db,
                "test_file": test_file
            }

@patch("src.rag.vector_store.Chroma")
@patch("src.rag.vector_store.OllamaEmbeddings")
@patch("src.rag.vector_store.SemanticChunker")
def test_full_ingestion_lifecycle(mock_chunker_class, mock_embeddings, mock_chroma, tmp_path):
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
    
    mock_chunker = MagicMock()
    mock_chunker_class.return_value = mock_chunker
    # Return a single chunk with parent metadata
    mock_chunker.split_documents.side_effect = lambda docs: [
        Document(page_content=d.page_content, metadata=d.metadata) for d in docs
    ]

    # Mock PROCESSED_DATA_DIR to be our test directory
    mock_processed_dir = MagicMock(spec=Path)
    mock_processed_dir.exists.return_value = True
    mock_processed_dir.glob.return_value = [test_file]
    mock_processed_dir.__str__.return_value = str(processed_dir)
    mock_processed_dir.joinpath.side_effect = lambda *args: processed_dir.joinpath(*args)

    # Patch module-level constants in vector_store.py
    with patch("src.rag.vector_store.PROCESSED_DATA_DIR", mock_processed_dir), \
         patch("src.rag.vector_store.CHROMA_DB_DIR", chroma_dir), \
         patch("src.rag.vector_store.HASH_DB_PATH", hash_db), \
         patch("src.rag.vector_store.PROJECT_ROOT", tmp_path):
        
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
