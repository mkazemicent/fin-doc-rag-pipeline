# Enterprise Deal Analytics & Risk Extraction Pipeline

A production-ready, agentic RAG pipeline designed to automate the extraction of critical financial terms and assess risks in unstructured corporate contracts (e.g., Syndicated Loans, ISDA Agreements). 

Built with strict data privacy considerations for the Canadian financial services sector.

## The Business Problem
Processing complex, unstructured deal documents manually is time-consuming and prone to human error. Missing a conflicting maturity date or a poorly structured liability clause introduces significant institutional risk. 

## The Solution
This pipeline utilizes an Agentic Retrieval-Augmented Generation (RAG) architecture to ingest PDF contracts, completely mask Personally Identifiable Information (PII), and deploy an autonomous AI agent to extract terms, flag missing clauses, and generate stakeholder-ready summaries. 

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

## Key Features
* **Compliance-First Ingestion:** All documents are passed through a zero-trust PII masking layer before any data is embedded or sent to an LLM.
* **Agentic Retrieval:** Utilizes LangGraph to give the AI specific "tools" (Term Extractor, Risk Assessor) and a self-correction loop.
* **Automated Risk Red-Flagging:** Automatically cross-references extracted dates and financial figures to highlight term mismatches.

## 🚀 To-Do / Roadmap
- [ ] **Hybrid Search:** Implement BM25 (keyword search) alongside Vector search to guarantee exact document ID retrieval.
- [ ] **Agentic / Semantic Chunking:** Replace fixed splitting with an NLP model to dynamically split chunks based on meaning.
- [X] **Role-Based Access Control (RBAC):** Implement security models for isolating deal access between analyst teams.
- [X] **Export to Memo:** Add a feature for analysts to download generated insights as structured memos.
- [ ] **Containerization:** Finalize Docker deployment for secure, isolated environments.

---
*Developed for Canadian Financial Services Compliance.*
