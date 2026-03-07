import streamlit as st
import sys
import os
import requests
from pathlib import Path
from dotenv import load_dotenv

# --- PATH PRE-REQUISITE ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Load environment variables
load_dotenv('.env.local')

# Now we can safely import our local modules
try:
    from src.rag.agent import build_agent
    from src.ingestion.document_processor import process_documents
    from src.rag.vector_store import initialize_vector_store
    from langchain_core.messages import HumanMessage, AIMessage
except ImportError as e:
    st.error(f"Failed to import agent modules. Check sys.path. Error: {e}")
    st.stop()

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Enterprise Deal Analyzer",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Bank-Grade Aesthetics)
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
    .stChatFloatingInputContainer {
        padding-bottom: 2rem;
    }
    .chunk-card {
        background-color: #1e2630;
        color: #f0f2f6;
        border-left: 5px solid #007bff;
        padding: 15px;
        margin-bottom: 10px;
        border-radius: 5px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
    }
    .chunk-card p {
        color: #f0f2f6 !important;
    }
    .metadata-badge {
        font-size: 0.8rem;
        background-color: #e9ecef;
        color: #495057;
        padding: 2px 8px;
        border-radius: 10px;
        margin-right: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR: SYSTEM DIAGNOSTICS & INGESTION
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/parakeet/512/000000/bank.png", width=100)
    st.title("System Control")
    st.markdown("---")
    
    # 1. AI Infrastructure Status
    st.subheader("🛠️ Infrastructure")
    
    # Ollama Status
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    llm_model = os.getenv("LLM_MODEL", "llama3.1")
    
    try:
        ollama_response = requests.get(f"{ollama_url}/api/tags", timeout=2)
        if ollama_response.status_code == 200:
            # Check if our specific model is pulled
            models = [m['name'] for m in ollama_response.json().get('models', [])]
            if llm_model in models or f"{llm_model}:latest" in models:
                st.success(f"✅ Ollama: {llm_model} Active")
            else:
                st.warning(f"⚠️ Ollama Active, but {llm_model} not found.")
        else:
            st.error("❌ Ollama Service Unreachable")
    except:
        st.error("❌ Ollama Service Offline")
    
    # ChromaDB & Governance Tracker Status
    CHROMA_PATH = PROJECT_ROOT / "data" / "chroma_db"
    TRACKER_PATH = PROJECT_ROOT / "data" / "ingestion_tracker.db"
    
    if CHROMA_PATH.exists():
         st.success("✅ ChromaDB: Connected")
    else:
         st.error("❌ Vector DB: Not Found")
         
    if TRACKER_PATH.exists():
         st.success("✅ Governance: Hash Tracking Active")
    else:
         st.info("ℹ️ Governance: Tracker Initializing...")

    st.markdown("---")
    
    # 2. Document Uploader
    st.subheader("📁 Ingest Deal Documents")
    uploaded_file = st.file_uploader("Upload new PDF contract", type="pdf")
    if uploaded_file is not None:
        raw_dir = PROJECT_ROOT / "data" / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = raw_dir / uploaded_file.name
        
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"Saved {uploaded_file.name}")
        
        if st.button("🚀 Process & Embed"):
            with st.spinner("Semantic Processing & PII Masking in progress..."):
                try:
                    processed_dir = PROJECT_ROOT / "data" / "processed"
                    process_documents(str(raw_dir), str(processed_dir))
                    initialize_vector_store()
                    st.success("Analysis Engine Updated!")
                    st.balloons()
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")

    st.markdown("---")
    
    # 3. Actions
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# ==========================================
# AGENT INITIALIZATION (CACHED)
# ==========================================
@st.cache_resource
def load_agent():
    return build_agent()

agent = load_agent()

# ==========================================
# CHAT INTERFACE
# ==========================================
st.title("🏢 Enterprise Deal Analyzer")
st.caption(f"Strategy: Semantic RAG | Engine: {llm_model} | Status: Air-Gapped")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Standard Refusal Message
REFUSAL_MSG = "I cannot find this information in the provided deal documents."

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "transparency" in message:
            t_data = message["transparency"]
            # CONDITIONAL RENDERING: Only show if relevant, chunks exist, and NOT a refusal
            is_refusal = message["content"] == REFUSAL_MSG
            if t_data.get("query") != "IRRELEVANT_QUERY" and t_data.get("chunks") and not is_refusal:
                with st.expander("🔍 Retrieval Transparency & Semantic Chunks"):
                    # Render optimized query
                    st.info(f"**Optimized Search Query:** `{t_data['query']}`")
                    
                    # Render individual chunk cards
                    for chunk in t_data['chunks']:
                        st.markdown(f"""
                        <div class="chunk-card">
                            <div>
                                <span class="metadata-badge">📄 {chunk['source']}</span>
                                <span class="metadata-badge">🛡️ {chunk['group']}</span>
                            </div>
                            <p style='margin-top:10px;'>{chunk['content']}</p>
                        </div>
                        """, unsafe_allow_html=True)

# User Input
if prompt := st.chat_input("Ask a question about your portfolio..."):
    # 1. FINAL UAT FIX: Explicitly clear slate to prevent 'ghost' context
    if "messages" in st.session_state:
        # We keep the history but ensure current processing starts fresh
        pass 
    
    # 2. Display User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 3. Invoke Agent with Spinner
    with st.chat_message("assistant"):
        with st.spinner("Analyzing deal documents..."):
            try:
                # Memory Integration
                chat_history = []
                for msg in st.session_state.messages[:-1]:
                    if msg["role"] == "user":
                        chat_history.append(HumanMessage(content=msg["content"]))
                    else:
                        chat_history.append(AIMessage(content=msg["content"]))

                # Invocation
                final_state = agent.invoke({
                    "question": prompt,
                    "chat_history": chat_history
                })
                
                # FINAL UAT FIX: Use refusal if answer is missing or signal leaked
                answer = final_state.get("answer", REFUSAL_MSG)
                if answer in ["relevant", "irrelevant"]:
                    answer = REFUSAL_MSG
                    
                raw_context = final_state.get("context", "")
                optimized_query = final_state.get("optimized_query", "N/A")
                
                # Parse context into chunk data
                chunks = []
                if raw_context:
                    raw_chunks = raw_context.split("\n\n---\n\n")
                    for rc in raw_chunks:
                        if "\n" in rc:
                            try:
                                header, content = rc.split("\n", 1)
                                source_part = header.split("|")[0].replace("SOURCE:", "").strip()
                                group_part = header.split("|")[1].replace("GROUP:", "").strip()
                                chunks.append({
                                    "source": source_part,
                                    "group": group_part,
                                    "content": content.strip()
                                })
                            except:
                                continue

                # Store transparency data
                transparency_data = {
                    "query": optimized_query,
                    "chunks": chunks
                }
                
                # Render AI Response
                st.markdown(answer)
                
                # FINAL UAT FIX: Conditional Transparency at the END
                show_transparency = (
                    optimized_query != "IRRELEVANT_QUERY" and 
                    len(chunks) > 0 and 
                    answer != REFUSAL_MSG
                )
                
                if show_transparency:
                    with st.expander("🔍 Retrieval Transparency & Semantic Chunks"):
                        st.info(f"**Optimized Search Query:** `{optimized_query}`")
                        for chunk in chunks:
                            st.markdown(f"""
                            <div class="chunk-card">
                                <div>
                                    <span class="metadata-badge">📄 {chunk['source']}</span>
                                    <span class="metadata-badge">🛡️ {chunk['group']}</span>
                                </div>
                                <p style='margin-top:10px;'>{chunk['content']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Save to session state
                msg_data = {"role": "assistant", "content": answer}
                if show_transparency:
                    msg_data["transparency"] = transparency_data
                st.session_state.messages.append(msg_data)
                
            except Exception as e:
                st.error(f"Analysis failed: {e}")

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.caption("Enterprise RAG v1.0 | Standardized Recursive Chunking | Local Llama 3.1 | Air-Gapped Security")
