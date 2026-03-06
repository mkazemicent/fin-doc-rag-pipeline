import pytest
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
        "retry_count": 0
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
        "retry_count": 1 # Below the cap of 3
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
        "retry_count": 3 # Cap reached
    }
    
    next_node = decide_to_generate(state)
    assert next_node == "generate"

# ===========================================================================
# Category 2: Grading Node Logic (Isolated from Routing)
# ===========================================================================

@patch("src.rag.agent.ChatOllama")
def test_grading_node_yes(mock_llm_class):
    """Verify that 'yes' from LLM maps to 'relevant' signal."""
    # Setup mock
    mock_llm = MagicMock()
    mock_llm_class.return_value = mock_llm
    # LangChain chain invoke returns a string (parsed by StrOutputParser)
    mock_llm.invoke.return_value.content = "yes"
    
    state: AgentState = {
        "question": "What is the rate?",
        "context": "The rate is 5%.",
        "retry_count": 0
    }
    
    result = grade_context_node(state)
    
    assert result["answer"] == "relevant"
    # Success should NOT increment the counter
    assert "retry_count" not in result

@patch("src.rag.agent.ChatOllama")
def test_grading_node_no_increments_counter(mock_llm_class):
    """Verify that 'no' from LLM maps to 'irrelevant' and increments counter."""
    mock_llm = MagicMock()
    mock_llm_class.return_value = mock_llm
    mock_llm.invoke.return_value.content = "no"
    
    state: AgentState = {
        "question": "What is the rate?",
        "context": "Some irrelevant text.",
        "retry_count": 1
    }
    
    result = grade_context_node(state)
    
    assert result["answer"] == "irrelevant"
    assert result["retry_count"] == 2 # 1 + 1
