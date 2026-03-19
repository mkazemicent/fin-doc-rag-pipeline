from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from src.config import Settings
from src.rag.utils import size_cap_chunk


# ===========================================================================
# Category 1: Two-Stage Semantic Chunking Logic
# ===========================================================================

class TestTwoStageChunking:
    """Verify that two-stage chunking preserves section boundaries and applies size caps."""

    def test_small_sections_stay_intact(self):
        """Sections smaller than max_chunk_size should not be re-split."""
        text = "Section 1: Short content.\n\nSection 2: Also short."
        metadata = {"source": "deal.pdf", "access_group": "general"}

        semantic_texts = text.split("\n\n")
        chunks = size_cap_chunk(semantic_texts, metadata, max_chunk_size=1500, min_chunk_chars=0)

        assert len(chunks) == 2
        assert chunks[0].page_content == "Section 1: Short content."
        assert chunks[1].page_content == "Section 2: Also short."

    def test_oversized_section_gets_re_split(self):
        """Sections exceeding max_chunk_size should be split by RecursiveCharacterTextSplitter."""
        long_section = "Financial terms. " * 200  # ~3400 chars
        text = f"Short section.\n\n{long_section}"
        metadata = {"source": "deal.pdf", "access_group": "general"}

        semantic_texts = text.split("\n\n")
        chunks = size_cap_chunk(semantic_texts, metadata, max_chunk_size=1500, chunk_size=800, min_chunk_chars=0)

        # Short section stays intact
        assert chunks[0].page_content == "Short section."
        # Long section gets re-split into multiple chunks
        assert len(chunks) > 2

    def test_metadata_preserved_through_both_stages(self):
        """Metadata must be preserved in both small and re-split chunks."""
        long_section = "Covenant details. " * 150
        text = f"Short header.\n\n{long_section}"
        metadata = {"source": "contract.pdf", "access_group": "general"}

        semantic_texts = text.split("\n\n")
        chunks = size_cap_chunk(semantic_texts, metadata, max_chunk_size=1500)

        for chunk in chunks:
            assert chunk.metadata["source"] == "contract.pdf"
            assert chunk.metadata["access_group"] == "general"

    def test_empty_sections_filtered(self):
        """Empty sections (whitespace-only) should be filtered out."""
        text = "Section 1.\n\n   \n\nSection 2."
        metadata = {"source": "deal.pdf", "access_group": "general"}

        semantic_texts = text.split("\n\n")
        chunks = size_cap_chunk(semantic_texts, metadata, min_chunk_chars=0)

        assert len(chunks) == 2
        assert all(chunk.page_content.strip() for chunk in chunks)


# ===========================================================================
# Category 2: Semantic Chunker Fallback
# ===========================================================================

class TestSemanticChunkerFallback:
    """Verify that SemanticChunker failure falls back to size-based splitting."""

    @patch("src.rag.chroma_deal_store.Chroma")
    @patch("src.rag.chroma_deal_store.OllamaEmbeddings")
    @patch("src.rag.chroma_deal_store.chromadb")
    @patch("src.rag.chroma_deal_store.SemanticChunker")
    def test_semantic_chunker_failure_falls_back(
        self, mock_semantic_class, mock_chromadb, mock_embeddings, mock_chroma, tmp_path
    ):
        """If SemanticChunker raises, fallback to RecursiveCharacterTextSplitter."""
        from src.rag.chroma_deal_store import ChromaDealStore

        processed_dir = tmp_path / "processed"
        processed_dir.mkdir()
        (processed_dir / "deal.txt").write_text(
            "This credit agreement contains financial covenants and benchmark replacement provisions "
            "for Term CORRA and Daily Compounded CORRA as applicable under Canadian securities law."
        )

        test_settings = Settings(data_dir=tmp_path)

        mock_semantic = MagicMock()
        mock_semantic_class.return_value = mock_semantic
        mock_semantic.split_text.side_effect = Exception("Embedding service unavailable")

        mock_vs = MagicMock()
        mock_chroma.return_value = mock_vs

        with patch("src.rag.chroma_deal_store.IngestionTracker") as mock_tracker_class, \
             patch("src.rag.chroma_deal_store.TextLoader") as mock_loader:

            mock_tracker = MagicMock()
            mock_tracker_class.return_value = mock_tracker
            mock_tracker.__enter__ = MagicMock(return_value=mock_tracker)
            mock_tracker.__exit__ = MagicMock(return_value=False)
            mock_tracker.check_and_hash.return_value = (False, "fakehash123")

            mock_loader.return_value.load.return_value = [
                Document(
                    page_content=(
                        "This credit agreement contains financial covenants and benchmark replacement provisions "
                        "for Term CORRA and Daily Compounded CORRA as applicable under Canadian securities law."
                    ),
                    metadata={}
                )
            ]

            store = ChromaDealStore(settings=test_settings)
            store.initialize_deal_store()

            # Should still have called add_documents (fallback produced chunks)
            assert mock_vs.add_documents.called


# ===========================================================================
# Category 3: Tracker Reset
# ===========================================================================

class TestTrackerReset:
    """Verify that IngestionTracker.reset() clears all records."""

    def test_reset_clears_all_records(self, tmp_path):
        from src.ingestion.hash_tracker import IngestionTracker

        db_path = str(tmp_path / "test_tracker.db")
        tracker = IngestionTracker(db_path)

        # Create a dummy file and mark it processed
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        tracker.mark_as_processed(str(test_file))
        assert tracker.is_already_processed(str(test_file)) is True

        # Reset and verify
        tracker.reset()
        assert tracker.is_already_processed(str(test_file)) is False
        tracker.close()
