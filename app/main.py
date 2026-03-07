import streamlit as st
import sys
import requests
from pathlib import Path

# --- PATH PRE-REQUISITE ---
# app/main.py is located inside the 'app' directory.
# Resolve the project root to allow absolute imports from the 'src' package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

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
        background-color: #f8f9fa;
    }
    .stChatFloatingInputContainer {
        padding-bottom: 2rem;
    }
    .st-emotion-cache-1c7n2ka {
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR: SYSTEM DIAGNOSTICS
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/parakeet/512/000000/bank.png", width=100)
    st.title("System Diagnostics")
    st.markdown("---")
    
    # 1. Ollama Status
    try:
        ollama_response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if ollama_response.status_code == 200:
            st.success("✅ Local Ollama (Llama 3.1) Active")
        else:
            st.error("❌ Ollama Service Unreachable")
    except:
        st.error("❌ Ollama Service Unreachable")
    
    # 2. ChromaDB Status
    CHROMA_PATH = PROJECT_ROOT / "data" / "chroma_db"
    if CHROMA_PATH.exists():
         st.success("✅ ChromaDB Connected")
    else:
         st.warning("⚠️ Vector DB Not Found (re-ingest needed)")
         
    # 3. Presidio/SpaCy Status
    try:
        from presidio_analyzer import AnalyzerEngine
        st.success("✅ Presidio PII Masking Active")
    except Exception as e:
        st.error(f"❌ PII Masking Offline: {e}")
        
    st.markdown("---")
    
    # 4. Document Uploader
    st.subheader("📁 Upload Deal Documents")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        raw_dir = PROJECT_ROOT / "data" / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = raw_dir / uploaded_file.name
        
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"Saved {uploaded_file.name}")
        
        with st.spinner("Processing document and updating vector database..."):
            try:
                processed_dir = PROJECT_ROOT / "data" / "processed"
                process_documents(str(raw_dir), str(processed_dir))
                initialize_vector_store()
                st.success("Analysis engine updated with new document!")
            except Exception as e:
                st.error(f"Failed to process document: {e}")

    st.markdown("---")
    
    # 5. Export Feature
    def generate_memo():
        memo = "# Enterprise Deal Analysis Report\n\n"
        for msg in st.session_state.messages:
            role = "USER" if msg["role"] == "user" else "AI ANALYST"
            memo += f"### {role}\n{msg['content']}\n\n"
            if "citations" in msg:
                memo += f"#### Source Citations:\n{msg['citations']}\n\n"
        return memo

    if st.session_state.get("messages"):
        st.download_button(
            label="📥 Export Analysis to Memo",
            data=generate_memo(),
            file_name="deal_analysis_report.md",
            mime="text/markdown"
        )

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
st.caption("Strategic Portfolio Analysis & Financial Term Extraction")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # If there are citations, show them inside the expander
        if "citations" in message:
            with st.expander("🔍 Retrieval Transparency & Citations"):
                st.markdown(message["citations"])

# User Input
if prompt := st.chat_input("Ask a deal-specific question (e.g., 'What is the maturity date?')"):
    # 1. Display User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Invoke Agent with Spinner
    with st.chat_message("assistant"):
        with st.spinner("AI is analyzing deal documents..."):
            try:
                # Convert session history to LangChain messages for "Conversational Memory"
                chat_history = []
                for msg in st.session_state.messages[:-1]: # Exclude the current prompt
                    if msg["role"] == "user":
                        chat_history.append(HumanMessage(content=msg["content"]))
                    else:
                        chat_history.append(AIMessage(content=msg["content"]))

                # LangGraph Agent invocation
                final_state = agent.invoke({
                    "question": prompt,
                    "chat_history": chat_history
                })
                
                answer = final_state.get("answer", "I encountered an error while analyzing the deal.")
                context = final_state.get("context", "No context retrieved.")
                optimized_query = final_state.get("optimized_query", "N/A")
                
                # Format Citations (Cleaned Up)
                formatted_citations = f"**Optimized Search Query:** `{optimized_query}`\n\n---\n\n"
                # If the context is just one big string, we can split it or just display it
                # For Phase 5, we'll assume the context may contain markers or just display it clearly.
                formatted_citations += f"{context}"
                
                # Render AI Response
                st.markdown(answer)
                with st.expander("🔍 Retrieval Transparency & Citations"):
                    st.markdown(formatted_citations)
                
                # Save to session state
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "citations": formatted_citations
                })
                
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"ERROR: {e}"
                })
