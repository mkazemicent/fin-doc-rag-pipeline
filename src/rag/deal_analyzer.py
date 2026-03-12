import logging
from enum import Enum
from functools import partial
from typing import TypedDict, List, Optional, Union
from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from src.config import Settings, get_settings

logger = logging.getLogger(__name__)

# --- Query Status Enum ---
class QueryStatus(str, Enum):
    """Typed routing signals for the LangGraph state machine."""
    RELEVANT = "relevant"
    IRRELEVANT = "irrelevant"
    ERROR = "error"

# Sentinel token the LLM is instructed to output for non-financial queries.
IRRELEVANT_QUERY_TOKEN = "IRRELEVANT_QUERY"

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
    retrieved_docs: List[Document]
    routing_signal: str
    answer: Union[str, DealExtraction]
    retry_count: int
    chat_history: List[Union[HumanMessage, AIMessage]]
    user_role: str


# --- RBAC Role Mapping ---
ROLE_ACCESS_GROUPS: dict[str, list[str]] = {
    "general": ["general"],
    "compliance": ["general", "compliance"],
    "admin": ["general", "compliance", "confidential"],
}


# ==========================================================================
# NODE FUNCTIONS
# Each accepts optional injected dependencies.  When called standalone
# (e.g. unit-tests) they fall back to creating deps from settings.
# In the compiled graph, build_deal_analyzer() binds shared instances
# via functools.partial so the LLM and retriever are created only once.
# ==========================================================================

def rewrite_node(state: AgentState, *, llm=None) -> dict:
    """Node 1: Rewrites the user's prompt to be highly optimized for Vector Search."""
    logger.info("--- NODE: OPTIMIZING SEARCH QUERY ---")
    if llm is None:
        s = get_settings()
        llm = ChatOllama(model=s.llm_model, base_url=s.ollama_base_url, temperature=0)

    question = state["question"]
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are a strictly constrained data extraction script. Your ONLY job is to extract financial keywords from a user query to use in a vector database search.

        RULES:
        1. NEVER write conversational text.
        2. If the query is non-financial (recipe, weather, trivia), output exactly: {IRRELEVANT_QUERY_TOKEN}
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


def retrieve_node(state: AgentState, *, retriever=None, store=None, reranker=None) -> dict:
    """Node 2: Retrieves context using the OPTIMIZED query, then reranks against the original question."""
    logger.info("--- NODE: RETRIEVING CONTEXT ---")
    optimized_query = state["optimized_query"]

    if optimized_query == IRRELEVANT_QUERY_TOKEN:
        logger.warning("Irrelevant query detected. Skipping ChromaDB retrieval.")
        return {"context": "", "retrieved_docs": [], "routing_signal": QueryStatus.IRRELEVANT}

    # Retriever resolution: injected retriever (tests) > store with RBAC filter > fallback default
    if retriever is None:
        if store is None:
            from src.rag.chroma_deal_store import ChromaDealStore
            store = ChromaDealStore()
        user_role = state.get("user_role", "general")
        allowed_groups = ROLE_ACCESS_GROUPS.get(user_role, ["general"])
        retriever = store.get_retriever(where_filter={"access_group": {"$in": allowed_groups}})

    try:
        docs = retriever.invoke(optimized_query)
        logger.info(f"Retrieved {len(docs)} candidates via MMR.")

        # Rerank against the original question for precision
        if reranker and docs:
            docs = reranker.compress_documents(docs, state["question"])
            logger.info(f"Reranked to {len(docs)} top results.")

        context_list = []
        for doc in docs:
            source = doc.metadata.get("source", "Unknown")
            group = doc.metadata.get("access_group", "general")
            context_list.append(f"SOURCE: {source} | GROUP: {group}\n{doc.page_content}")

        context = "\n\n---\n\n".join(context_list)
        return {"context": context, "retrieved_docs": docs}
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        return {"context": "", "retrieved_docs": [], "routing_signal": QueryStatus.ERROR}


def grade_context_node(state: AgentState, *, llm=None, max_retries=3) -> dict:
    """Node 3: Evaluates if the retrieved context is relevant."""
    logger.info("--- NODE: GRADING RETRIEVED CONTEXT ---")
    context = state.get("context", "")
    retry_count = state.get("retry_count", 0)
    routing_signal = state.get("routing_signal", "")

    if routing_signal in (QueryStatus.IRRELEVANT, QueryStatus.ERROR) or not context:
        return {"routing_signal": QueryStatus.IRRELEVANT, "retry_count": max_retries}

    if llm is None:
        s = get_settings()
        llm = ChatOllama(model=s.llm_model, base_url=s.ollama_base_url, temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Evaluate if the provided context is relevant to the question. Output 'yes' or 'no'."),
        ("human", "Question: {question} \n\nContext: {context}")
    ])

    chain = prompt | llm | StrOutputParser()
    score = chain.invoke({"question": state["question"], "context": context}).strip().lower()

    if "yes" in score:
        return {"routing_signal": QueryStatus.RELEVANT}
    else:
        return {"routing_signal": QueryStatus.IRRELEVANT, "retry_count": retry_count + 1}


def generate_node(state: AgentState, *, llm=None) -> dict:
    """Node 4: Generates the structured DealExtraction answer or a refusal."""
    logger.info("--- NODE: GENERATING ANSWER ---")
    question = state["question"]
    context = state.get("context", "")
    routing_signal = state.get("routing_signal", "")

    if routing_signal == QueryStatus.IRRELEVANT:
        return {"answer": "I am a strictly air-gapped financial assistant designed to analyze Canadian corporate contracts. I cannot fulfill non-financial requests such as recipes, code generation, or general trivia."}

    if routing_signal == QueryStatus.ERROR or not context:
        return {"answer": "I cannot find this information in the provided deal documents."}

    if llm is None:
        s = get_settings()
        llm = ChatOllama(model=s.llm_model, base_url=s.ollama_base_url, temperature=0)

    structured_llm = llm.with_structured_output(DealExtraction)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Senior Deal Desk Analyst. Extract precise financial terms from corporate contracts. "
                   "Your goal is to fill the DealExtraction schema based ONLY on the provided context. "
                   "If a field is missing in the text, provide a 'Not found in document' value. \n\nContext:\n{context}"),
        ("placeholder", "{chat_history}"),
        ("human", "{question}")
    ])

    chain = prompt | structured_llm
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


def decide_to_generate(state: AgentState, *, max_retries=3) -> str:
    """Routing Function: decides whether to generate or retry the query."""
    signal = state.get("routing_signal", QueryStatus.IRRELEVANT)
    retry_count = state.get("retry_count", 0)

    if signal == QueryStatus.RELEVANT or retry_count >= max_retries:
        return "generate"
    return "rewrite"


# ==========================================================================
# GRAPH BUILDER
# ==========================================================================

def build_deal_analyzer(settings: Optional[Settings] = None) -> CompiledStateGraph:
    """Builds the Deal Analytics LangGraph with injected dependencies."""
    settings = settings or get_settings()
    logger.info("Building Deal Analyzer State Machine...")

    # --- Shared dependencies (created ONCE, bound via partial) ---
    llm = ChatOllama(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
        temperature=0
    )

    from src.rag.chroma_deal_store import ChromaDealStore
    store_obj = ChromaDealStore(settings=settings)

    # FlashRank cross-encoder reranker (CPU-only, zero VRAM)
    from langchain_community.document_compressors import FlashrankRerank
    reranker = FlashrankRerank(
        model=settings.reranker_model,
        top_n=settings.rerank_top_n,
    )

    max_retries = settings.max_retries

    # --- GRAPH ASSEMBLY ---
    workflow = StateGraph(AgentState)

    workflow.add_node("rewrite", partial(rewrite_node, llm=llm))
    workflow.add_node("retrieve", partial(retrieve_node, store=store_obj, reranker=reranker))
    workflow.add_node("grade_context", partial(grade_context_node, llm=llm, max_retries=max_retries))
    workflow.add_node("generate", partial(generate_node, llm=llm))

    workflow.add_edge(START, "rewrite")
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("retrieve", "grade_context")

    workflow.add_conditional_edges(
        "grade_context",
        partial(decide_to_generate, max_retries=max_retries),
        {
            "rewrite": "rewrite",
            "generate": "generate"
        }
    )

    workflow.add_edge("generate", END)
    return workflow.compile()
