import streamlit as st
import sys
import os
import requests
import tempfile
import shutil
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
    from src.rag.deal_analyzer import build_deal_analyzer, DealExtraction
    from src.ingestion.document_processor import process_documents
    from src.rag.chroma_deal_store import initialize_vector_store
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
    .deal-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin-bottom: 20px;
    }
    .term-item {
        border-left: 3px solid #28a745;
        padding-left: 10px;
        margin-bottom: 5px;
    }
    .risk-item {
        border-left: 3px solid #dc3545;
        padding-left: 10px;
        margin-bottom: 5px;
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
            models = [m['name'] for m in ollama_response.json().get('models', [])]
            if llm_model in models or f"{llm_model}:latest" in models:
                st.success(f"✅ Ollama: {llm_model} Active")
            else:
                st.warning(f"⚠️ Ollama Active, but {llm_model} not found.")
        else:
            st.error("❌ Ollama Service Unreachable")
    except:
        st.error("❌ Ollama Service Offline")
    
    # ChromaDB (Server Connection Check)
    chroma_host = os.getenv("CHROMA_HOST", "localhost")
    chroma_port = os.getenv("CHROMA_PORT", "8000")
    # Using /api/v2/heartbeat as /api/v1/ is deprecated in latest Chroma
    chroma_url = f"http://{chroma_host}:{chroma_port}/api/v2/heartbeat"
    
    try:
        chroma_response = requests.get(chroma_url, timeout=2)
        if chroma_response.status_code == 200:
            st.success(f"✅ ChromaDB: Connected ({chroma_host})")
        else:
            # Fallback check for / if v2 is somehow not yet ready
            st.warning(f"⚠️ ChromaDB: Response {chroma_response.status_code}")
    except:
        st.error("❌ ChromaDB: Offline/Unreachable")
         
    # Governance Tracker
    TRACKER_PATH = PROJECT_ROOT / "data" / "ingestion_state.db"
    if TRACKER_PATH.exists():
         st.success("✅ Governance: Tracking Active")
    else:
         st.info("ℹ️ Governance: Initializing...")

    st.markdown("---")
    
    # 2. Document Uploader
    st.subheader("📁 Ingest Deal Documents")
    uploaded_file = st.file_uploader("Upload new PDF contract", type="pdf")
    
    if uploaded_file is not None:
        raw_dir = PROJECT_ROOT / "data" / "raw"
        tmp_path = None
        
        try:
            os.makedirs(raw_dir, exist_ok=True)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name
            
            final_pdf_path = raw_dir / uploaded_file.name
            shutil.move(tmp_path, final_pdf_path)
            st.success(f"✅ Securely saved: {uploaded_file.name}")
            
            if st.button("🚀 Process & Embed"):
                with st.spinner("Processing & PII Masking..."):
                    try:
                        processed_dir = PROJECT_ROOT / "data" / "processed"
                        os.makedirs(processed_dir, exist_ok=True)
                        process_documents(str(raw_dir), str(processed_dir))
                        initialize_vector_store()
                        st.success("Analysis Engine Updated!")
                        st.balloons()
                    except Exception as e:
                        st.error(f"Ingestion failed: {e}")
        except PermissionError:
            st.error("🚨 Permission Denied: UID 1000 cannot write to volume.")
        except Exception as e:
            st.error(f"Upload failed: {e}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    st.markdown("---")
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# ==========================================
# AGENT INITIALIZATION (CACHED)
# ==========================================
@st.cache_resource
def load_agent():
    return build_deal_analyzer()

agent = load_agent()

# ==========================================
# UTILS
# ==========================================
def render_deal_extraction(extraction: DealExtraction):
    """Renders the DealExtraction Pydantic model."""
    with st.container():
        st.markdown(f"### 📅 Maturity Date: **{extraction.maturity_date}**")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📝 Key Deal Terms")
            for term in extraction.deal_terms:
                st.markdown(f"<div class='term-item'>{term}</div>", unsafe_allow_html=True)
        with col2:
            st.markdown("#### ⚠️ Risk Factors")
            for risk in extraction.risk_factors:
                st.markdown(f"<div class='risk-item'>{risk}</div>", unsafe_allow_html=True)

# ==========================================
# CHAT INTERFACE
# ==========================================
st.title("🏢 Enterprise Deal Analyzer")
st.caption(f"Strategy: Semantic RAG | Engine: {llm_model} | Status: Air-Gapped")

if "messages" not in st.session_state:
    st.session_state.messages = []

REFUSAL_MSG = "I cannot find this information in the provided deal documents."

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if isinstance(message["content"], DealExtraction):
            render_deal_extraction(message["content"])
        else:
            st.markdown(message["content"])
        if "transparency" in message:
            t_data = message["transparency"]
            if t_data.get("query") != "IRRELEVANT_QUERY" and t_data.get("chunks"):
                with st.expander("🔍 Retrieval Transparency"):
                    st.info(f"**Optimized Query:** `{t_data['query']}`")
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

if prompt := st.chat_input("Ask a question about your portfolio..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                chat_history = []
                for msg in st.session_state.messages[:-1]:
                    content = str(msg["content"])
                    if msg["role"] == "user":
                        chat_history.append(HumanMessage(content=content))
                    else:
                        chat_history.append(AIMessage(content=content))

                final_state = agent.invoke({
                    "question": prompt,
                    "chat_history": chat_history
                })
                
                answer = final_state.get("answer", REFUSAL_MSG)
                if answer in ["relevant", "irrelevant"]:
                    answer = REFUSAL_MSG
                    
                raw_context = final_state.get("context", "")
                optimized_query = final_state.get("optimized_query", "N/A")
                
                chunks = []
                if raw_context:
                    raw_chunks = raw_context.split("\n\n---\n\n")
                    for rc in raw_chunks:
                        if "\n" in rc:
                            try:
                                header, content = rc.split("\n", 1)
                                source_part = header.split("|")[0].replace("SOURCE:", "").strip()
                                group_part = header.split("|")[1].replace("GROUP:", "").strip()
                                chunks.append({"source": source_part, "group": group_part, "content": content.strip()})
                            except: continue

                transparency_data = {"query": optimized_query, "chunks": chunks}
                
                if isinstance(answer, DealExtraction):
                    render_deal_extraction(answer)
                else:
                    st.markdown(answer)
                
                show_transparency = (optimized_query != "IRRELEVANT_QUERY" and len(chunks) > 0 and answer != REFUSAL_MSG)
                if show_transparency:
                    with st.expander("🔍 Retrieval Transparency"):
                        st.info(f"**Optimized Query:** `{optimized_query}`")
                        for chunk in chunks:
                            st.markdown(f"""<div class="chunk-card"><div><span class="metadata-badge">📄 {chunk['source']}</span><span class="metadata-badge">🛡️ {chunk['group']}</span></div><p style='margin-top:10px;'>{chunk['content']}</p></div>""", unsafe_allow_html=True)
                
                msg_data = {"role": "assistant", "content": answer}
                if show_transparency: msg_data["transparency"] = transparency_data
                st.session_state.messages.append(msg_data)
            except Exception as e:
                st.error(f"Analysis failed: {e}")

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.caption("Enterprise Deal Analyzer v2.0 | Client-Server RAG | Air-Gapped Security")
