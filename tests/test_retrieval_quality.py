from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from src.config import Settings
from src.rag.deal_analyzer import retrieve_node, AgentState, IRRELEVANT_QUERY_TOKEN, QueryStatus


# ===========================================================================
# Category 1: Reranker Integration
# ===========================================================================

class TestRerankerIntegration:
    """Verify that the reranker filters and reorders retrieved documents."""

    def test_reranker_compresses_results(self):
        """Reranker should reduce candidate count to top_n."""
        mock_retriever = MagicMock()
        candidates = [
            Document(page_content=f"Financial covenant clause {i}: This section describes credit terms and benchmark replacement provisions applicable to the Borrower under the credit agreement dated June 2024.", metadata={"source": "deal.pdf", "access_group": "general"})
            for i in range(20)
        ]
        mock_retriever.invoke.return_value = candidates

        mock_reranker = MagicMock()
        top_results = candidates[:5]
        mock_reranker.compress_documents.return_value = top_results

        state: AgentState = {
            "question": "What is the maturity date?",
            "optimized_query": "maturity date deal",
            "context": "",
            "routing_signal": "",
            "retry_count": 0,
            "chat_history": []
        }

        result = retrieve_node(state, retriever=mock_retriever, reranker=mock_reranker)

        mock_reranker.compress_documents.assert_called_once()
        # Verify reranker was called with original question, not optimized query
        call_args = mock_reranker.compress_documents.call_args
        assert call_args[0][1] == "What is the maturity date?"
        # Context should be built from the 5 reranked results
        assert result["context"].count("SOURCE:") == 5

    def test_retrieve_without_reranker_returns_all(self):
        """When no reranker is provided, all retrieved docs should be returned."""
        mock_retriever = MagicMock()
        docs = [
            Document(page_content=f"Financial clause {i}: The Borrower shall maintain a minimum Interest Coverage Ratio of not less than 1.25 to 1.00 calculated on a rolling twelve-month basis as of each Quarterly Reporting Date pursuant to Section 6.1.", metadata={"source": "deal.pdf", "access_group": "general"})
            for i in range(8)
        ]
        mock_retriever.invoke.return_value = docs

        state: AgentState = {
            "question": "What is the rate?",
            "optimized_query": "interest rate",
            "context": "",
            "routing_signal": "",
            "retry_count": 0,
            "chat_history": []
        }

        result = retrieve_node(state, retriever=mock_retriever, reranker=None)
        assert result["context"].count("SOURCE:") == 8

    def test_irrelevant_query_skips_retriever_and_reranker(self):
        """IRRELEVANT_QUERY token should bypass both retriever and reranker."""
        mock_retriever = MagicMock()
        mock_reranker = MagicMock()

        state: AgentState = {
            "question": "How to make pasta?",
            "optimized_query": IRRELEVANT_QUERY_TOKEN,
            "context": "",
            "routing_signal": "",
            "retry_count": 0,
            "chat_history": []
        }

        result = retrieve_node(state, retriever=mock_retriever, reranker=mock_reranker)

        mock_retriever.invoke.assert_not_called()
        mock_reranker.compress_documents.assert_not_called()
        assert result["routing_signal"] == QueryStatus.IRRELEVANT

    def test_reranker_with_empty_results(self):
        """Reranker should not be called when retriever returns no documents."""
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []

        mock_reranker = MagicMock()

        state: AgentState = {
            "question": "What is the rate?",
            "optimized_query": "interest rate",
            "context": "",
            "routing_signal": "",
            "retry_count": 0,
            "chat_history": []
        }

        retrieve_node(state, retriever=mock_retriever, reranker=mock_reranker)
        mock_reranker.compress_documents.assert_not_called()


# ===========================================================================
# Category 2: MMR Retriever Configuration
# ===========================================================================

class TestMMRRetrieverConfig:
    """Verify that the retriever is configured with MMR search."""

    @patch("src.rag.chroma_deal_store.Chroma")
    @patch("src.rag.chroma_deal_store.OllamaEmbeddings")
    @patch("src.rag.chroma_deal_store.chromadb")
    def test_retriever_uses_mmr(self, mock_chromadb, mock_embeddings, mock_chroma):
        """Verify get_retriever produces an MMR retriever with correct params."""
        from src.rag.chroma_deal_store import ChromaDealStore

        test_settings = Settings(
            retriever_k=20,
            mmr_lambda=0.7,
        )

        mock_vs = MagicMock()
        mock_chroma.return_value = mock_vs

        store = ChromaDealStore(settings=test_settings)
        store.get_retriever()

        mock_vs.as_retriever.assert_called_once_with(
            search_type="mmr",
            search_kwargs={
                "k": 20,
                "fetch_k": 200,
                "lambda_mult": 0.7,
            }
        )

    @patch("src.rag.chroma_deal_store.Chroma")
    @patch("src.rag.chroma_deal_store.OllamaEmbeddings")
    @patch("src.rag.chroma_deal_store.chromadb")
    def test_retriever_custom_k(self, mock_chromadb, mock_embeddings, mock_chroma):
        """Verify get_retriever accepts a custom k override."""
        from src.rag.chroma_deal_store import ChromaDealStore

        test_settings = Settings(retriever_k=20, mmr_lambda=0.7)

        mock_vs = MagicMock()
        mock_chroma.return_value = mock_vs

        store = ChromaDealStore(settings=test_settings)
        store.get_retriever(k=10)

        call_kwargs = mock_vs.as_retriever.call_args[1]
        assert call_kwargs["search_kwargs"]["k"] == 10
        assert call_kwargs["search_kwargs"]["fetch_k"] == 100

    @patch("src.rag.chroma_deal_store.Chroma")
    @patch("src.rag.chroma_deal_store.OllamaEmbeddings")
    @patch("src.rag.chroma_deal_store.chromadb")
    def test_retriever_passes_rbac_filter(self, mock_chromadb, mock_embeddings, mock_chroma):
        """Verify that a provided filter is passed into search_kwargs."""
        from src.rag.chroma_deal_store import ChromaDealStore

        test_settings = Settings(retriever_k=20, mmr_lambda=0.7)
        mock_vs = MagicMock()
        mock_chroma.return_value = mock_vs

        store = ChromaDealStore(settings=test_settings)
        rbac_filter = {"access_group": {"$in": ["general", "compliance"]}}
        store.get_retriever(where_filter=rbac_filter)

        call_kwargs = mock_vs.as_retriever.call_args[1]
        assert call_kwargs["search_kwargs"]["filter"] == {"access_group": {"$in": ["general", "compliance"]}}
