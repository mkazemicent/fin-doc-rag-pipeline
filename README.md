# Enterprise Deal Analytics & Risk Extraction Pipeline

A production-ready, agentic RAG pipeline designed to automate the extraction of critical financial terms and assess risks in unstructured corporate contracts (e.g., Syndicated Loans, ISDA Agreements). 

Built with strict data privacy considerations for the Canadian financial services sector.

## The Business Problem
Processing complex, unstructured deal documents manually is time-consuming and prone to human error. Missing a conflicting maturity date or a poorly structured liability clause introduces significant institutional risk. 

## The Solution
This pipeline utilizes an Agentic Retrieval-Augmented Generation (RAG) architecture to ingest PDF contracts, completely mask Personally Identifiable Information (PII), and deploy an autonomous AI agent to extract terms, flag missing clauses, and generate stakeholder-ready summaries. 

## Key Capabilities
* **Compliance-First Ingestion:** All documents are passed through a zero-trust PII masking layer before any data is embedded or sent to an LLM.
* **Agentic Retrieval:** Utilizes LangGraph to give the AI specific "tools" (Term Extractor, Risk Assessor) and a self-correction loop.
* **Automated Risk Red-Flagging:** Automatically cross-references extracted dates and financial figures to highlight term mismatches.

## Core Architecture
```mermaid
graph TD
    subgraph "Data Ingestion & Privacy Layer"
        A[Raw SEDAR+ PDFs] --> B(PyPDFLoader)
        B --> C{Microsoft Presidio}
        C -- Detects PII --> D[Mask Entities: Email, Phone, IBAN]
        D --> E[Processed TXT Files]
    end

    subgraph "Air-Gapped Vector Pipeline"
        E --> F(Recursive Text Splitter)
        F -- Chunks 500/50 --> G(Ollama: mxbai-embed-large)
        G --> H[(Local ChromaDB)]
    end

    subgraph "Application Layer"
        I[Streamlit UI] <--> J((LangGraph Agent))
        J <--> |Search Query| H
        J <--> |Context & Prompt| K(Ollama: Llama 3.1 8B)
    end
```

## Advanced Agentic Features

### 🔄 Dynamic Routing & Self-Correction
This pipeline implements an advanced **LangGraph State Machine** to ensure high-quality information retrieval. If the initial search results are deemed irrelevant by the **Context Grader**, the agent automatically rewrites the search query and attempts a more focused retrieval (up to 3 times).

```mermaid
graph TD
    START((Start)) --> REWRITE[Node: Rewrite Query]
    REWRITE --> RETRIEVE[Node: Retrieve Context]
    RETRIEVE --> GRADE{Node: Grade Context}
    
    GRADE -- "Relevant / Max Retries reached" --> GENERATE[Node: Generate Answer]
    GRADE -- "Irrelevant (Retry < 3)" --> REWRITE
    
    GENERATE --> END((End))

    style GRADE fill:#f9f,stroke:#333,stroke-width:2px
    style REWRITE fill:#bbf,stroke:#333,stroke-width:1px
```

### 📊 Automated Evaluation (RAGAS)
To maintain the highest standards of accuracy for financial document analysis, we utilize the **RAGAS** framework. This allows us to mathematically track:
*   **Faithfulness:** Does the answer only use information from the retrieved context?
*   **Answer Relevancy:** How pertinent is the answer to the user's original query?

```mermaid
graph LR
    subgraph "Evaluation Workflow"
        Q[Test Questions] --> AGENT[LangGraph Agent]
        AGENT --> A[Answer]
        AGENT --> C[Context]
        
        A & C & Q --> RAGAS[RAGAS Evaluator]
        RAGAS --> REP[Scorecard / CSV Report]
    end
```

## 🛠️ Getting Started

### 1. Prerequisites
*   **Python:** 3.10 or higher.
*   **Local LLM:** [Ollama](https://ollama.com/) must be installed and running.
*   **Models:** Pull the required models before running:
    ```bash
    ollama pull llama3.1
    ollama pull mxbai-embed-large
    ```

### 2. Installation
```bash
git clone https://github.com/mkazemicent/fin-doc-rag-pipeline.git
cd fin-doc-rag-pipeline

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

## 🚀 Execution Guide (What to Run)

### **A. Full UI Experience (Recommended)**
Launch the bank-grade dashboard to upload documents and chat with the agent.
```bash
streamlit run app/main.py
```

### **B. Data Ingestion Pipeline**
If you have new PDFs in `data/raw`, run these in order to process and embed them (though the UI handles this automatically).
```bash
# 1. Mask PII and extract text
python -m src.ingestion.document_processor

# 2. Chunk and embed into Vector DB
python -m src.rag.vector_store
```

### **C. Evaluation & Testing**
Run mathematical evaluations or verify agent routing logic.
```bash
# Run RAGAS scorecard (Context Precision/Faithfulness)
python scripts/evaluate_ragas.py

# Live test of the LangGraph state machine routing
python test_live_routing.py
```

## 📂 Project Structure (What is What)
*   `app/main.py`: The Streamlit frontend and system diagnostics dashboard.
*   `src/ingestion/`:
    *   `document_processor.py`: Text extraction and Microsoft Presidio PII masking.
    *   `hash_tracker.py`: SQL-based incremental sync logic (avoids reprocessing unchanged files).
*   `src/rag/`:
    *   `agent.py`: The core **LangGraph** state machine and autonomous agent logic.
    *   `vector_store.py`: ChromaDB management and local embedding generation.
*   `data/`:
    *   `raw/`: Put your original PDF contracts here.
    *   `processed/`: Contains PII-masked `.txt` versions for embedding.
    *   `chroma_db/`: Persistent local vector storage.
*   `scripts/evaluate_ragas.py`: Automated performance reporting script.

## 🚀 To-Do / Roadmap
- [ ] **Hybrid Search:** Implement BM25 (keyword search) alongside Vector search to guarantee exact document ID retrieval.
- [ ] **Agentic / Semantic Chunking:** Replace fixed splitting with an NLP model to dynamically split chunks based on meaning.
- [X] **Role-Based Access Control (RBAC):** Implement security models for isolating deal access between analyst teams.
- [X] **Export to Memo:** Add a feature for analysts to download generated insights as structured memos.
- [ ] **Containerization:** Finalize Docker deployment for secure, isolated environments.

---
*Developed for Canadian Financial Services Compliance.*
