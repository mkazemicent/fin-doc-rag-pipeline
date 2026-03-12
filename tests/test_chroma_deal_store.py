from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from src.config import Settings
from src.rag.chroma_deal_store import ChromaDealStore
from src.rag.utils import size_cap_chunk

# ===========================================================================
# Category 1 — Metadata Integrity
# ===========================================================================

class TestMetadataIntegrity:
    """Validate that size_cap_chunk preserves critical metadata through both stages."""

    def test_metadata_preservation_small_sections(self):
        """Metadata must survive pass-through for sections under max_chunk_size."""
        metadata = {"source": "deal_v1.pdf", "access_group": "general"}
        semantic_texts = ["Short section about credit terms."]

        chunks = size_cap_chunk(semantic_texts, metadata, max_chunk_size=1500)

        assert len(chunks) == 1
        assert chunks[0].metadata["source"] == "deal_v1.pdf"
        assert chunks[0].metadata["access_group"] == "general"

    def test_metadata_preservation_oversized_sections(self):
        """Metadata must survive re-splitting for sections exceeding max_chunk_size."""
        metadata = {"source": "deal_v1.pdf", "access_group": "compliance"}
        long_text = "Financial covenant details. " * 200  # ~5400 chars
        semantic_texts = [long_text]

        chunks = size_cap_chunk(
            semantic_texts, metadata, max_chunk_size=1500, chunk_size=800
        )

        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.metadata["source"] == "deal_v1.pdf"
            assert chunk.metadata["access_group"] == "compliance"

# ===========================================================================
# Category 2 — Batch Processing Fault Tolerance
# ===========================================================================

class TestBatchProcessingResilience:
    """Verify the batch loop is fault-tolerant."""

    @patch("src.rag.chroma_deal_store.Chroma")
    def test_batch_processing_continues_after_partial_failure(self, mock_chroma, tmp_path):
        """
        Simulate a failure in one batch and success in others.
        With per-file tracking, a partial batch failure means the file
        is NOT marked as processed — it will be retried on next run.
        """
        # Create actual file structure
        processed_dir = tmp_path / "processed"
        processed_dir.mkdir()
        (processed_dir / "doc1.txt").write_text("test content")

        test_settings = Settings(data_dir=tmp_path)

        with patch("src.rag.chroma_deal_store.IngestionTracker") as mock_tracker_class, \
             patch("src.rag.chroma_deal_store.TextLoader") as mock_loader, \
             patch("src.rag.chroma_deal_store.OllamaEmbeddings"), \
             patch("src.rag.chroma_deal_store.SemanticChunker") as mock_semantic_class, \
             patch("src.rag.utils.RecursiveCharacterTextSplitter") as mock_splitter_class, \
             patch("src.rag.chroma_deal_store.chromadb"):

            mock_tracker = MagicMock()
            mock_tracker_class.return_value = mock_tracker
            mock_tracker.__enter__ = MagicMock(return_value=mock_tracker)
            mock_tracker.__exit__ = MagicMock(return_value=False)
            mock_tracker.check_and_hash.return_value = (False, "fakehash123")

            mock_loader.return_value.load.return_value = [Document(page_content="test", metadata={})]

            # SemanticChunker returns a single oversized text to trigger size-cap fallback
            mock_semantic = MagicMock()
            mock_semantic_class.return_value = mock_semantic
            mock_semantic.split_text.return_value = ["x" * 2000]

            mock_splitter = MagicMock()
            mock_splitter_class.return_value = mock_splitter
            fake_chunks = [Document(page_content=f"c{i}", metadata={}) for i in range(120)]
            mock_splitter.split_documents.return_value = fake_chunks

            mock_vs = MagicMock()
            mock_chroma.return_value = mock_vs
            mock_vs.add_documents.side_effect = [
                Exception("Timeout"), # Batch 1 fails
                None,                  # Batch 2 succeeds
                None                   # Batch 3 succeeds
            ]

            store = ChromaDealStore(settings=test_settings)
            store.initialize_deal_store()

            assert mock_vs.add_documents.call_count == 3
            # File should NOT be marked as processed due to batch failure
            assert not mock_tracker.mark_as_processed_with_hash.called

# ===========================================================================
# Category 3 — Settings Injection Validation
# ===========================================================================

def test_vector_store_uses_settings():
    """Verify that ChromaDealStore pulls model and URL from injected Settings."""
    test_settings = Settings(
        embedding_model="test-embed",
        ollama_base_url="http://test-ollama:11434"
    )

    with patch("src.rag.chroma_deal_store.OllamaEmbeddings") as mock_embeddings, \
         patch("src.rag.chroma_deal_store.chromadb"):
        ChromaDealStore(settings=test_settings)

        mock_embeddings.assert_called_once_with(
            model="test-embed",
            base_url="http://test-ollama:11434"
        )
