import pytest
import os
from unittest.mock import MagicMock, patch

from src.rag.deal_analyzer import decide_to_generate, grade_context_node, AgentState

# ===========================================================================
# Category 1: Deterministic Routing Logic
# ===========================================================================

def test_routing_to_generate_on_relevance():
    """If signal is 'relevant', we must route to 'generate'."""
    state: AgentState = {
        "question": "When is the deal?",
        "optimized_query": "deal maturity",
        "context": "The deal matures in 2025.",
        "answer": "relevant", 
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
        "retry_count": 1,
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
        "retry_count": 3, 
    }
    
    next_node = decide_to_generate(state)
    assert next_node == "generate"

# ===========================================================================
# Category 2: Environmental Config & Mocks
# ===========================================================================

@patch("src.rag.deal_analyzer.ChatOllama")
def test_agent_nodes_use_env_variables(mock_chat_ollama):
    """Verify that agent nodes pull LLM_MODEL and URL from .env.local."""
    with patch.dict(os.environ, {
        "LLM_MODEL": "test-llm",
        "OLLAMA_BASE_URL": "http://test-ollama:11434"
    }):
        state: AgentState = {"question": "test", "chat_history": []}
        
        from src.rag.deal_analyzer import rewrite_node
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
    """Verify that 'yes' from LLM maps to 'relevant' signal."""
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
        assert result["answer"] == "relevant"

@patch("src.rag.deal_analyzer.ChatOllama")
def test_grading_node_no_increments_counter(mock_llm_class):
    """Verify that 'no' from LLM maps to 'irrelevant' and increments counter."""
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
        assert result["answer"] == "irrelevant"
        assert result["retry_count"] == 2
