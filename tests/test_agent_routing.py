import pytest
from unittest.mock import MagicMock, patch

from src.config import Settings
from src.rag.deal_analyzer import decide_to_generate, grade_context_node, rewrite_node, AgentState

# ===========================================================================
# Category 1: Deterministic Routing Logic
# ===========================================================================

def test_routing_to_generate_on_relevance():
    """If routing_signal is 'relevant', we must route to 'generate'."""
    state: AgentState = {
        "question": "When is the deal?",
        "optimized_query": "deal maturity",
        "context": "The deal matures in 2025.",
        "routing_signal": "relevant",
        "retry_count": 0,
        "chat_history": []
    }

    next_node = decide_to_generate(state)
    assert next_node == "generate"

def test_routing_to_rewrite_on_irrelevance():
    """If routing_signal is 'irrelevant' and count < 3, we must route back to 'rewrite'."""
    state: AgentState = {
        "question": "When is the deal?",
        "optimized_query": "deal maturity",
        "context": "This is a recipe for apple pie.",
        "routing_signal": "irrelevant",
        "retry_count": 1,
        "chat_history": []
    }

    next_node = decide_to_generate(state)
    assert next_node == "rewrite"

def test_cap_at_max_retries():
    """If routing_signal is 'irrelevant' but count is 3, we must proceed to 'generate' to exit loop."""
    state: AgentState = {
        "question": "When is the deal?",
        "optimized_query": "deal maturity",
        "context": "Irrelevant noise.",
        "routing_signal": "irrelevant",
        "retry_count": 3,
    }

    next_node = decide_to_generate(state)
    assert next_node == "generate"

# ===========================================================================
# Category 2: Environmental Config & Mocks
# ===========================================================================

@patch("src.rag.deal_analyzer.ChatOllama")
def test_agent_nodes_use_settings(mock_chat_ollama):
    """Verify that agent nodes pull LLM_MODEL and URL from Settings."""
    test_settings = Settings(
        llm_model="test-llm",
        ollama_base_url="http://test-ollama:11434"
    )

    with patch("src.rag.deal_analyzer.get_settings", return_value=test_settings):
        state: AgentState = {"question": "test", "chat_history": []}

        try:
            rewrite_node(state)
        except Exception:
            pass

        mock_chat_ollama.assert_called_with(
            model="test-llm",
            base_url="http://test-ollama:11434",
            temperature=0
        )

@patch("src.rag.deal_analyzer.ChatOllama")
def test_grading_node_yes(mock_llm_class):
    """Verify that 'yes' from LLM maps to 'relevant' routing_signal."""
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "yes"
    mock_chain.__or__.return_value = mock_chain

    with patch("src.rag.deal_analyzer.ChatPromptTemplate.from_messages", return_value=mock_chain):
        state: AgentState = {
            "question": "What is the rate?",
            "context": "The rate is 5%.",
            "retry_count": 0,
            "chat_history": []
        }

        result = grade_context_node(state)
        assert result["routing_signal"] == "relevant"

@patch("src.rag.deal_analyzer.ChatOllama")
def test_grading_node_no_increments_counter(mock_llm_class):
    """Verify that 'no' from LLM maps to 'irrelevant' routing_signal and increments counter."""
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "no"
    mock_chain.__or__.return_value = mock_chain

    with patch("src.rag.deal_analyzer.ChatPromptTemplate.from_messages", return_value=mock_chain):
        state: AgentState = {
            "question": "What is the rate?",
            "context": "Some irrelevant text.",
            "retry_count": 1,
            "chat_history": []
        }

        result = grade_context_node(state)
        assert result["routing_signal"] == "irrelevant"
        assert result["retry_count"] == 2
