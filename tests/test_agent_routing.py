import pytest
import os
from unittest.mock import MagicMock, patch

from src.rag.agent import decide_to_generate, grade_context_node, AgentState

# ===========================================================================
# Category 1: Deterministic Routing Logic
# ===========================================================================

def test_routing_to_generate_on_relevance():
    """If signal is 'relevant', we must route to 'generate'."""
    state: AgentState = {
        "question": "When is the deal?",
        "optimized_query": "deal maturity",
        "context": "The deal matures in 2025.",
        "answer": "relevant", # Set by the grading node
        "retry_count": 0,
        "chat_history": []
    }
    
    next_node = decide_to_generate(state)
    assert next_node == "generate"

def test_routing_to_rewrite_on_irrelevance():
    """If signal is 'irrelevant' and count < 3, we must route back to 'rewrite'."""
    state: AgentState = {
        "question": "When is the deal?",
        "optimized_query": "deal maturity",
        "context": "This is a recipe for apple pie.",
        "answer": "irrelevant",
        "retry_count": 1, # Below the cap of 3
        "chat_history": []
    }
    
    next_node = decide_to_generate(state)
    assert next_node == "rewrite"

def test_cap_at_max_retries():
    """If signal is 'irrelevant' but count is 3, we must proceed to 'generate' to exit loop."""
    state: AgentState = {
        "question": "When is the deal?",
        "optimized_query": "deal maturity",
        "context": "Irrelevant noise.",
        "answer": "irrelevant",
        "retry_count": 3, # Cap reached
        "chat_history": []
    }
    
    next_node = decide_to_generate(state)
    assert next_node == "generate"

# ===========================================================================
# Category 2: Environmental Config & Mocks
# ===========================================================================

@patch("src.rag.agent.ChatOllama")
def test_agent_nodes_use_env_variables(mock_chat_ollama):
    """Verify that agent nodes pull LLM_MODEL and URL from .env.local."""
    with patch.dict(os.environ, {
        "LLM_MODEL": "test-llm",
        "OLLAMA_BASE_URL": "http://test-ollama:11434"
    }):
        # Mocking for rewrite_node
        state: AgentState = {"question": "test", "chat_history": []}
        
        # Test rewrite_node instantiation
        from src.rag.agent import rewrite_node
        try:
            rewrite_node(state)
        except Exception:
            pass # We expect failure because we aren't mocking the chain invocation, but we only care about the call to ChatOllama
            
        # Verify ChatOllama was called with env vars
        mock_chat_ollama.assert_called_with(
            model="test-llm",
            base_url="http://test-ollama:11434",
            temperature=0
        )

@patch("src.rag.agent.ChatOllama")
def test_grading_node_yes(mock_llm_class):
    """Verify that 'yes' from LLM maps to 'relevant' signal."""
    # Mock the chain created by prompt | llm | parser
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "yes"
    # Ensure the chain stays the same mock when piped (prompt | llm | parser)
    mock_chain.__or__.return_value = mock_chain
    
    with patch("src.rag.agent.ChatPromptTemplate.from_messages", return_value=mock_chain):
        state: AgentState = {
            "question": "What is the rate?",
            "context": "The rate is 5%.",
            "retry_count": 0,
            "chat_history": []
        }
        
        result = grade_context_node(state)
        assert result["answer"] == "relevant"
        assert "retry_count" not in result

@patch("src.rag.agent.ChatOllama")
def test_grading_node_no_increments_counter(mock_llm_class):
    """Verify that 'no' from LLM maps to 'irrelevant' and increments counter."""
    # Mock the chain
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "no"
    # Ensure the chain stays the same mock when piped
    mock_chain.__or__.return_value = mock_chain
    
    with patch("src.rag.agent.ChatPromptTemplate.from_messages", return_value=mock_chain):
        state: AgentState = {
            "question": "What is the rate?",
            "context": "Some irrelevant text.",
            "retry_count": 1,
            "chat_history": []
        }
        
        result = grade_context_node(state)
        assert result["answer"] == "irrelevant"
        assert result["retry_count"] == 2 # 1 + 1
