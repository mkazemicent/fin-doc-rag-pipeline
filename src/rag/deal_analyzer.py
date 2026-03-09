import os
import logging
from typing import TypedDict, List, Optional, Union
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from src.rag.chroma_deal_store import ChromaDealStore

# Load environment variables
load_dotenv('.env.local')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Structured Output Schema ---
class DealExtraction(BaseModel):
    """Structured extraction of financial deal details from corporate contracts."""
    deal_terms: List[str] = Field(description="Key financial terms, covenants, and conditions of the deal.")
    risk_factors: List[str] = Field(description="Potential risks, liabilities, or negative conditions identified.")
    maturity_date: str = Field(description="The date when the deal, agreement, or financial instrument expires or matures.")

# --- Agent State ---
class AgentState(TypedDict):
    question: str
    optimized_query: str
    context: str
    answer: Union[str, DealExtraction]
    retry_count: int
    chat_history: list

# --- Helper ---
def get_retriever():
    store = ChromaDealStore()
    return store.get_retriever(k=8)

# --- NODES ---

def rewrite_node(state: AgentState):
    """Node 1: Rewrites the user's prompt to be highly optimized for Vector Search."""
    logger.info("--- NODE: OPTIMIZING SEARCH QUERY ---")
    question = state["question"]
    llm = ChatOllama(
        model=os.getenv("LLM_MODEL", "llama3.1"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a strictly constrained data extraction script. Your ONLY job is to extract financial keywords from a user query to use in a vector database search.
        
        RULES:
        1. NEVER write conversational text.
        2. If the query is non-financial (recipe, weather, trivia), output exactly: IRRELEVANT_QUERY
        3. Use chat history to resolve references.
        4. Output ONLY the raw keywords separated by spaces.
        """),
        ("placeholder", "{chat_history}"),
        ("human", "Optimize this question: {question}")
    ])
    chain = prompt | llm | StrOutputParser()
    optimized = chain.invoke({
        "question": question,
        "chat_history": state.get("chat_history", [])
    }).strip()
    
    if not optimized:
        optimized = question
    
    logger.info(f"Optimized Query : {optimized}")
    return {"optimized_query": optimized}

def retrieve_node(state: AgentState):
    """Node 2: Retrieves context using the OPTIMIZED query."""
    logger.info("--- NODE: RETRIEVING CONTEXT ---")
    optimized_query = state["optimized_query"]
    
    # --- PERFORMANCE & SAFETY SHORT-CIRCUIT ---
    if optimized_query == "IRRELEVANT_QUERY":
        logger.warning("Irrelevant query detected. Skipping ChromaDB retrieval.")
        return {"context": "IRRELEVANT_CONTEXT"}
    
    try:
        retriever = get_retriever()
        docs = retriever.invoke(optimized_query)
        
        context_list = []
        for doc in docs:
            source = doc.metadata.get("source", "Unknown")
            group = doc.metadata.get("access_group", "general")
            context_list.append(f"SOURCE: {source} | GROUP: {group}\n{doc.page_content}")
        
        context = "\n\n---\n\n".join(context_list)
        return {"context": context}
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        return {"context": "ERROR_DURING_RETRIEVAL"}

def grade_context_node(state: AgentState):
    """Node 4: Evaluates if the retrieved context is relevant."""
    logger.info("--- NODE: GRADING RETRIEVED CONTEXT ---")
    optimized_query = state.get("optimized_query", "")
    context = state.get("context", "")
    retry_count = state.get("retry_count", 0)

    # Short-circuit if context is irrelevant or empty
    if optimized_query == "IRRELEVANT_QUERY" or not context or context == "IRRELEVANT_CONTEXT":
        return {"answer": "irrelevant", "retry_count": 3} # Force exit to generation

    llm = ChatOllama(
        model=os.getenv("LLM_MODEL", "llama3.1"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Evaluate if the provided context is relevant to the question. Output 'yes' or 'no'."),
        ("human", "Question: {question} \n\nContext: {context}")
    ])

    chain = prompt | llm | StrOutputParser()
    score = chain.invoke({"question": state["question"], "context": context}).strip().lower()

    if "yes" in score:
        return {"answer": "relevant"}
    else:
        return {"answer": "irrelevant", "retry_count": retry_count + 1}

def generate_node(state: AgentState):
    """Node 3: Generates the structured DealExtraction answer or a refusal."""
    logger.info("--- NODE: GENERATING ANSWER ---")
    question = state["question"]
    context = state["context"]
    optimized_query = state.get("optimized_query", "")
    
    # 1. Handle Irrelevant Requests or Errors Early
    if optimized_query == "IRRELEVANT_QUERY" or context == "IRRELEVANT_CONTEXT":
        return {"answer": "I am a strictly air-gapped financial assistant designed to analyze Canadian corporate contracts. I cannot fulfill non-financial requests such as recipes, code generation, or general trivia."}

    if context == "ERROR_DURING_RETRIEVAL" or not context:
        return {"answer": "I cannot find this information in the provided deal documents."}

    # 2. Proceed with Structured Extraction for relevant financial queries
    llm = ChatOllama(
        model=os.getenv("LLM_MODEL", "llama3.1"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0
    ).with_structured_output(DealExtraction)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Senior Deal Desk Analyst. Extract precise financial terms from corporate contracts. "
                   "Your goal is to fill the DealExtraction schema based ONLY on the provided context. "
                   "If a field is missing in the text, provide a 'Not found in document' value. \n\nContext:\n{context}"),
        ("placeholder", "{chat_history}"),
        ("human", "{question}")
    ])
    
    chain = prompt | llm
    try:
        structured_result = chain.invoke({
            "context": context, 
            "question": question,
            "chat_history": state.get("chat_history", [])
        })
        return {"answer": structured_result}
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return {"answer": "Failed to generate structured extraction for this document."}

def decide_to_generate(state: AgentState):
    """Routing Function."""
    signal = state.get("answer", "irrelevant")
    retry_count = state.get("retry_count", 0)

    if signal == "relevant" or retry_count >= 3:
        return "generate"
    return "rewrite"

def build_deal_analyzer():
    """Builds the Deal Analytics LangGraph."""
    logger.info("Building Deal Analyzer State Machine...")
    workflow = StateGraph(AgentState)
    
    workflow.add_node("rewrite", rewrite_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_context", grade_context_node)
    workflow.add_node("generate", generate_node)
    
    workflow.add_edge(START, "rewrite")
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("retrieve", "grade_context")
    
    workflow.add_conditional_edges(
        "grade_context",
        decide_to_generate,
        {
            "rewrite": "rewrite",
            "generate": "generate"
        }
    )
    
    workflow.add_edge("generate", END)
    return workflow.compile()
