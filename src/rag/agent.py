import logging
from typing import TypedDict
from pathlib import Path

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 1. UPDATED STATE: We added an 'optimized_query' variable
class AgentState(TypedDict):
    question: str
    optimized_query: str
    context: str
    answer: str
    retry_count: int
    chat_history: list  # --- NEW: Stores LangChain messages ---

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHROMA_DB_DIR = PROJECT_ROOT / "data" / "chroma_db"

def get_retriever():
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    vectorstore = Chroma(persist_directory=str(CHROMA_DB_DIR), embedding_function=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 20})

# 2. THE NEW REWRITE NODE
def rewrite_node(state: AgentState):
    """Node 1: Rewrites the user's prompt to be highly optimized for Vector Search."""
    logger.info("--- NODE: OPTIMIZING SEARCH QUERY ---")
    question = state["question"]
    llm = ChatOllama(model="llama3.1", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a strictly constrained data extraction script. Your ONLY job is to extract financial keywords from a user query to use in a vector database search.
        
        RULES:
        1. NEVER write conversational text (e.g., "Here is the query", "I am an expert").
        2. If the user asks a non-financial question (like recipes or weather), output exactly the word: IRRELEVANT_QUERY
        3. Use the provided chat history to resolve references (like 'it', 'above', or 'that agreement').
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
    
    # --- SENIOR ENGINEER FALLBACK ---
    if not optimized:
        logger.warning("LLM failed to generate an optimized query. Falling back to original prompt.")
        optimized = question
    # -------------------------------
    
    logger.info(f"Original Prompt : {question}")
    logger.info(f"Optimized Query : {optimized}")
    return {"optimized_query": optimized}

def retrieve_node(state: AgentState):
    """Node 2: Retrieves context using the OPTIMIZED query, not the raw question."""
    logger.info("--- NODE: RETRIEVING CONTEXT ---")
    optimized_query = state["optimized_query"]  # Use the AI-generated query!
    
    retriever = get_retriever()
    docs = retriever.invoke(optimized_query)
    
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Keep the debugging file so we can verify the improvement
    with open("debug_context.txt", "w", encoding="utf-8") as f:
        f.write(context)
        
    return {"context": context}

def generate_node(state: AgentState):
    """Node 3: Generates the final answer."""
    logger.info("--- NODE: GENERATING ANSWER ---")
    question = state["question"]
    context = state["context"]
    
    llm = ChatOllama(model="llama3.1", temperature=0) 
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Senior Deal Desk Analyst at a major Canadian Bank. "
                   "Your job is to read unstructured corporate contracts and extract precise financial terms. "
                   "Answer the user's question based ONLY on the following context. "
                   "Incorporate the ongoing conversation history for continuity. "
                   "If the answer is not in the context, explicitly state: 'I cannot find this information in the provided deal documents.' "
                   "Do not guess. Do not hallucinate. \n\nContext:\n{context}"),
        ("placeholder", "{chat_history}"),
        ("human", "{question}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "context": context, 
        "question": question,
        "chat_history": state.get("chat_history", [])
    })
    return {"answer": answer}

# --- NEW NODES & ROUTING FOR PHASE 4 ---

def grade_context_node(state: AgentState):
    """Node 4: Evaluates if the retrieved context is relevant to the question."""
    logger.info("--- NODE: GRADING RETRIEVED CONTEXT ---")
    question = state["question"]
    context = state["context"]
    retry_count = state.get("retry_count", 0)

    llm = ChatOllama(model="llama3.1", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a quality control agent for a RAG pipeline. "
                   "Your job is to evaluate if the provided context is relevant to the user's question. "
                   "If the context contains information that helps answer the question, output 'yes'. "
                   "If the context is irrelevant or lacks the necessary details, output 'no'. "
                   "Output ONLY the word 'yes' or 'no'. No explanation."),
        ("human", "Question: {question} \n\nContext: {context}")
    ])

    chain = prompt | llm | StrOutputParser()
    score = chain.invoke({"question": question, "context": context}).strip().lower()

    if "yes" in score:
        logger.info(f"Context Grade: RELEVANT (Loop count: {retry_count})")
        return {"answer": "relevant"} # We use 'answer' field as a temporary signal
    else:
        logger.warning(f"Context Grade: IRRELEVANT (Loop count: {retry_count})")
        return {"answer": "irrelevant", "retry_count": retry_count + 1}

def decide_to_generate(state: AgentState):
    """Routing Function: Determines whether to try again or generate the final answer."""
    logger.info("--- ROUTING: DECIDING NEXT STEP ---")
    
    # We use the signal passed in the 'answer' field from grade_context_node
    signal = state.get("answer", "irrelevant")
    retry_count = state.get("retry_count", 0)

    if signal == "relevant":
        logger.info("Decision: Proceed to GENERATE.")
        return "generate"
    
    if retry_count < 3:
        logger.warning(f"Decision: REWRITE query (Attempt {retry_count}/3).")
        return "rewrite"
    else:
        logger.error("Decision: MAX RETRIES REACHED. Proceeding to generate failure message.")
        return "generate"

def build_agent():
    logger.info("Building LangGraph State Machine with Self-Correction Loop...")
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("rewrite", rewrite_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_context", grade_context_node) # --- NEW ---
    workflow.add_node("generate", generate_node)
    
    # Define the sophisticated flow
    workflow.add_edge(START, "rewrite")
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("retrieve", "grade_context") # --- NEW: Intercept before generation ---
    
    # --- DYNAMIC ROUTING ---
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

if __name__ == "__main__":
    logger.info("Initializing Enterprise Deal Analyzer Agent...")
    app = build_agent()
    
    test_question = "What is the maturity date for the Cheeb Royalties Limited Partnership credit agreement?"
    logger.info(f"USER PROMPT: {test_question}")
    
    result = app.invoke({"question": test_question})
    
    logger.info("=====================================================")
    logger.info("FINAL AI ANALYSIS:")
    logger.info(result["answer"])
    logger.info("=====================================================")