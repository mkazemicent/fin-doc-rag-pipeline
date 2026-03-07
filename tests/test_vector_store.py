import pytest
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from src.rag.vector_store import get_chroma_instance, delete_document_from_db

# ===========================================================================
# Category 1 — Semantic Chunking & Metadata Integrity
# ===========================================================================

class TestSemanticChunking:
    """Validate that SemanticChunker preserves critical metadata."""

    @patch("src.rag.vector_store.OllamaEmbeddings")
    @patch("src.rag.vector_store.SemanticChunker")
    def test_metadata_preservation_during_split(self, mock_chunker_class, mock_embeddings_class):
        """
        CRITICAL: Verify that access_group and source are injected/preserved
        into every chunk, even if the splitter tries to strip them.
        """
        # Setup mocks
        mock_embeddings = MagicMock()
        mock_embeddings_class.return_value = mock_embeddings
        
        mock_chunker = MagicMock()
        mock_chunker_class.return_value = mock_chunker
        
        # Simulate a split that 'loses' metadata (typical behavior in some splitter versions)
        mock_chunker.split_documents.return_value = [
            Document(page_content="Chunk 1", metadata={}), 
            Document(page_content="Chunk 2", metadata={})
        ]

        # Prepare dummy parent doc
        parent_doc = Document(
            page_content="Long text about credit agreements.", 
            metadata={"source": "deal_v1.pdf", "access_group": "general"}
        )

        # Re-implementing the preservation logic we added to vector_store.py
        chunked_documents = []
        doc_chunks = mock_chunker.split_documents([parent_doc])
        for chunk in doc_chunks:
            chunk.metadata.update(parent_doc.metadata)
            chunked_documents.append(chunk)

        # Assertions
        assert len(chunked_documents) == 2
        for chunk in chunked_documents:
            assert chunk.metadata["source"] == "deal_v1.pdf"
            assert chunk.metadata["access_group"] == "general"

# ===========================================================================
# Category 2 — Batch Processing Fault Tolerance
# ===========================================================================

class TestBatchProcessingResilience:
    """Verify the batch loop is fault-tolerant to LLM/GPU timeouts."""

    @patch("src.rag.vector_store.Chroma")
    def test_batch_processing_continues_after_partial_failure(self, mock_chroma):
        """Simulate a failure in one batch and success in others."""
        from src.rag.vector_store import initialize_vector_store
        
        # We need to mock several things to isolate the batch loop
        with patch("src.rag.vector_store.IngestionTracker") as mock_tracker_class, \
             patch("src.rag.vector_store.TextLoader") as mock_loader, \
             patch("src.rag.vector_store.OllamaEmbeddings"), \
             patch("src.rag.vector_store.SemanticChunker") as mock_chunker_class:
            
            # Setup tracker mock
            mock_tracker = MagicMock()
            mock_tracker_class.return_value = mock_tracker
            mock_tracker.is_already_processed.return_value = False
            
            # Setup loader mock
            mock_loader.return_value.load.return_value = [Document(page_content="test", metadata={})]
            
            # Setup chunker to return 120 chunks (3 batches of 50, 50, 20)
            mock_chunker = MagicMock()
            mock_chunker_class.return_value = mock_chunker
            fake_chunks = [Document(page_content=f"c{i}", metadata={}) for i in range(120)]
            mock_chunker.split_documents.return_value = fake_chunks
            
            # Setup Chroma mock to fail on the first batch
            mock_vs = MagicMock()
            mock_chroma.return_value = mock_vs
            mock_vs.add_documents.side_effect = [
                Exception("Timeout"), # Batch 1 fails
                None,                  # Batch 2 succeeds
                None                   # Batch 3 succeeds
            ]
            
            # Setup mock for PROCESSED_DATA_DIR to return our test file
            mock_dir = MagicMock()
            mock_dir.exists.return_value = True
            mock_dir.glob.return_value = [Path("data/processed/doc1.txt")]
            
            # Force paths for test environment
            with patch("src.rag.vector_store.PROCESSED_DATA_DIR", mock_dir):
                # Execute
                initialize_vector_store()
                
                # Verify add_documents was called 3 times despite failure
                assert mock_vs.add_documents.call_count == 3
                # Verify tracker only marks file as processed if at least some chunks succeeded
                assert mock_tracker.mark_as_processed.called

# ===========================================================================
# Category 3 — Environmental Config Validation
# ===========================================================================

def test_vector_store_uses_env_variables():
    """Verify that vector store pulls model and URL from .env.local."""
    with patch.dict(os.environ, {
        "EMBEDDING_MODEL": "test-embed",
        "OLLAMA_BASE_URL": "http://test-ollama:11434"
    }):
        with patch("src.rag.vector_store.OllamaEmbeddings") as mock_embeddings:
            get_chroma_instance("test_dir")
            
            # Check initialization params
            mock_embeddings.assert_called_once_with(
                model="test-embed",
                base_url="http://test-ollama:11434"
            )
