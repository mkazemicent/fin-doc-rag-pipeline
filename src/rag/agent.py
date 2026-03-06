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
        ("system", "You are an expert search query optimizer for a Vector Database. "
                   "Convert the user's question into a highly focused, keyword-dense search query. "
                   "Extract only the core financial terms and the specific company. "
                   "IMPORTANT: Output ONLY the optimized keywords. Do not include any conversational text."),
        ("human", "Optimize this question: {question}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    optimized = chain.invoke({"question": question}).strip()
    
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
                   "If the answer is not in the context, explicitly state: 'I cannot find this information in the provided deal documents.' "
                   "Do not guess. Do not hallucinate. \n\nContext:\n{context}"),
        ("human", "{question}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    return {"answer": answer}

def build_agent():
    logger.info("Building LangGraph State Machine...")
    workflow = StateGraph(AgentState)
    
    # Add our 3 nodes
    workflow.add_node("rewrite", rewrite_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    
    # Define the new flow
    workflow.add_edge(START, "rewrite")
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("retrieve", "generate")
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