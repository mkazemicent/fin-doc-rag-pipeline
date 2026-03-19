import streamlit as st
import sys
import os
import logging
import requests
import tempfile
import shutil
from html import escape
from pathlib import Path

# --- PATH PRE-REQUISITE ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Centralized logging (single entry point)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Now we can safely import our local modules
try:
    from src.config import get_settings
    from src.rag.deal_analyzer import build_deal_analyzer, DealExtraction, IRRELEVANT_QUERY_TOKEN
    from src.rag.utils import evaluate_show_transparency, serialize_for_history, should_render_transparency
    from src.ingestion.document_processor import process_documents
    from src.rag.chroma_deal_store import ChromaDealStore
    from langchain_core.messages import HumanMessage, AIMessage
except ImportError as e:
    st.error(f"Failed to import agent modules. Check sys.path. Error: {e}")
    st.stop()

settings = get_settings()

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
# AGENT INITIALIZATION (CACHED)
# ==========================================
@st.cache_resource
def load_agent():
    return build_deal_analyzer()

@st.cache_resource
def load_masker():
    from src.ingestion.document_processor import PIIMasker
    return PIIMasker(model_name="en_core_web_lg")

agent = load_agent()

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
    ollama_url = settings.ollama_base_url
    llm_model = settings.llm_model
    
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
    except Exception:
        st.error("❌ Ollama Service Offline")
    
    # ChromaDB (Server Connection Check)
    chroma_host = settings.chroma_host
    chroma_port = settings.chroma_port
    # Using /api/v2/heartbeat as /api/v1/ is deprecated in latest Chroma
    chroma_url = f"http://{chroma_host}:{chroma_port}/api/v2/heartbeat"
    
    try:
        chroma_response = requests.get(chroma_url, timeout=2)
        if chroma_response.status_code == 200:
            st.success(f"✅ ChromaDB: Connected ({chroma_host})")
        else:
            # Fallback check for / if v2 is somehow not yet ready
            st.warning(f"⚠️ ChromaDB: Response {chroma_response.status_code}")
    except Exception:
        st.error("❌ ChromaDB: Offline/Unreachable")
         
    # Governance Tracker
    TRACKER_PATH = settings.hash_db_path
    if TRACKER_PATH.exists():
         st.success("✅ Governance: Tracking Active")
    else:
         st.info("ℹ️ Governance: Initializing...")

    st.markdown("---")

    # 2. User Role (RBAC)
    st.subheader("🔐 User Role")
    user_role = st.selectbox("Select role:", ["general", "compliance", "admin"], index=0)

    st.markdown("---")

    # 3. Document Uploader
    st.subheader("📁 Ingest Deal Documents")
    uploaded_file = st.file_uploader("Upload contract (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
    
    if uploaded_file is not None:
        raw_dir = settings.data_root / "raw"
        tmp_path = None

        try:
            os.makedirs(raw_dir, exist_ok=True)
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name

            final_pdf_path = raw_dir / uploaded_file.name
            shutil.move(tmp_path, final_pdf_path)
            st.success(f"✅ Securely saved: {uploaded_file.name}")

            access_group = st.selectbox("Access Level:", ["general", "compliance", "confidential"])

            if st.button("🚀 Process & Embed"):
                progress_bar = st.progress(0, text="Processing documents...")
                try:
                    processed_dir = settings.processed_data_dir
                    os.makedirs(processed_dir, exist_ok=True)

                    def _update_progress(completed, total):
                        progress_bar.progress(
                            completed / total,
                            text=f"Processed {completed}/{total} documents..."
                        )

                    process_documents(
                        str(raw_dir), str(processed_dir),
                        settings=settings, masker=load_masker(),
                        progress_callback=_update_progress,
                    )
                    progress_bar.progress(1.0, text="Embedding & storing...")
                    ChromaDealStore(settings=settings).initialize_deal_store(access_group=access_group)
                    progress_bar.empty()
                    st.success("Analysis Engine Updated!")
                    st.balloons()
                except Exception as e:
                    progress_bar.empty()
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
# UTILS
# ==========================================
_ACCESS_BADGE_STYLE: dict[str, str] = {
    "general":      "background:#e9ecef;color:#495057",
    "compliance":   "background:#fff3cd;color:#856404",
    "confidential": "background:#f8d7da;color:#842029",
}

def _badge_style(access_group: str) -> str:
    return _ACCESS_BADGE_STYLE.get(access_group, _ACCESS_BADGE_STYLE["general"])


def render_deal_extraction(extraction: DealExtraction):
    """Renders the DealExtraction Pydantic model."""
    with st.container():
        st.markdown(f"### 📅 Maturity Date: **{escape(extraction.maturity_date)}**")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📝 Key Deal Terms")
            for term in extraction.deal_terms:
                st.markdown(f"<div class='term-item'>{escape(term)}</div>", unsafe_allow_html=True)
        with col2:
            st.markdown("#### ⚠️ Risk Factors")
            for risk in extraction.risk_factors:
                st.markdown(f"<div class='risk-item'>{escape(risk)}</div>", unsafe_allow_html=True)

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
        if should_render_transparency(message):
            t_data = message["transparency"]
            if t_data.get("query") != IRRELEVANT_QUERY_TOKEN and t_data.get("chunks"):
                with st.expander("🔍 Retrieval Transparency"):
                    st.info(f"**Optimized Query:** `{t_data['query']}`")
                    for chunk in t_data['chunks']:
                        ag = chunk.metadata.get('access_group', 'general')
                        st.markdown(f"""
                        <div class="chunk-card">
                            <div>
                                <span class="metadata-badge">📄 {escape(chunk.metadata.get('source', 'Unknown'))}</span>
                                <span class="metadata-badge" style="{_badge_style(ag)}">🛡️ {escape(ag)}</span>
                            </div>
                            <p style='margin-top:10px;'>{escape(chunk.page_content)}</p>
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
                    content = serialize_for_history(msg["content"])
                    if msg["role"] == "user":
                        chat_history.append(HumanMessage(content=content))
                    else:
                        chat_history.append(AIMessage(content=content))

                final_state = agent.invoke({
                    "question": prompt,
                    "chat_history": chat_history,
                    "user_role": user_role,
                })

                answer = final_state.get("answer", REFUSAL_MSG)
                routing_signal = final_state.get("routing_signal", "")

                optimized_query = final_state.get("optimized_query", "N/A")
                chunks = final_state.get("retrieved_docs", [])

                transparency_data = {"query": optimized_query, "chunks": chunks}
                
                if isinstance(answer, DealExtraction):
                    render_deal_extraction(answer)
                else:
                    st.markdown(answer)
                
                show_transparency = evaluate_show_transparency(routing_signal, optimized_query, chunks)
                if show_transparency:
                    with st.expander("🔍 Retrieval Transparency"):
                        st.info(f"**Optimized Query:** `{optimized_query}`")
                        for chunk in chunks:
                            ag = chunk.metadata.get('access_group', 'general')
                            st.markdown(f"""<div class="chunk-card"><div><span class="metadata-badge">📄 {escape(chunk.metadata.get('source', 'Unknown'))}</span><span class="metadata-badge" style="{_badge_style(ag)}">🛡️ {escape(ag)}</span></div><p style='margin-top:10px;'>{escape(chunk.page_content)}</p></div>""", unsafe_allow_html=True)
                
                msg_data = {"role": "assistant", "content": answer, "routing_signal": routing_signal}
                if show_transparency:
                    msg_data["transparency"] = transparency_data
                st.session_state.messages.append(msg_data)
            except Exception as e:
                st.error(f"Analysis failed: {e}")

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.caption("Enterprise Deal Analyzer v2.0 | Client-Server RAG | Air-Gapped Security")
