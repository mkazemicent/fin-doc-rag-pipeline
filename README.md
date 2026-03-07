# Enterprise Deal Analytics & Risk Extraction Pipeline (v1.0)

A production-ready, agentic RAG pipeline designed to automate the extraction of critical financial terms and assess risks in unstructured corporate contracts. 

Built with 100% air-gapped security and strict data governance for the financial services sector.

## Core Architecture
```mermaid
graph TD
    subgraph "Governance & Ingestion Layer"
        A[Raw PDF Contracts] --> B(Presidio PII Masking)
        B --> C{SQLite Hash Check}
        C -- "New/Modified" --> D[Semantic-Aware Recursive Split]
        C -- "Unchanged" --> SKIP[Skip Ingestion]
        D --> E[Inject Metadata: access_group]
    end

    subgraph "Air-Gapped Vector DB"
        E --> F(Ollama: mxbai-embed-large)
        F --> G[(Local ChromaDB)]
    end

    subgraph "Agentic Reasoning (LangGraph)"
        H[Streamlit UI] --> I(Rewrite Node)
        I --> J(Retrieve Node)
        J --> K{Grade Context Node}
        K -- "Relevant" --> L(Generate Node)
        K -- "Irrelevant < 3" --> I
        L --> H
    end
```

## Key Capabilities
* **Governance Layer:** Integrated **SQLite Hash Tracking** ensures incremental ingestion, skipping unchanged files to optimize local GPU resources.
* **Semantic-Aware Recursive Strategy:** Utilizes an optimized splitting logic (800/150) with paragraph and sentence-level awareness to ensure 100% reliability with local LLM context windows.
* **Agentic Self-Correction:** Implements a **LangGraph State Machine** that autonomously grades retrieval quality and rewrites queries until a relevant context is found.
* **RBAC Groundwork:** Every document chunk is injected with `access_group` metadata at the ingestion stage for future enterprise scaling.

## 📊 Quality & Validation
* **Test Coverage:** **71% Statement Coverage** achieved via a standardized `pytest` suite.
* **Reliability:** Heavy use of unit test mocks ensures deterministic validation of routing, PII masking, and metadata preservation without requiring a live GPU.
* **RAGAS Evaluation:** Mathematical scoring of Faithfulness and Relevancy using the RAGAS framework.

## 🛠️ Getting Started

### 1. Prerequisites
*   **Python:** 3.10 or higher.
*   **Local LLM:** [Ollama](https://ollama.com/) must be installed and running.
*   **Models:** `ollama pull llama3.1` and `ollama pull mxbai-embed-large`.

### 2. Installation
```bash
git clone https://github.com/mkazemicent/fin-doc-rag-pipeline.git
cd fin-doc-rag-pipeline

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies and setup environment
pip install -r requirements.txt
cp .env.example .env.local  # Update your local config
```

### 3. Running Tests
The project uses a standardized `pyproject.toml` configuration.
```bash
python -m pytest --cov=src tests/
```

## 🚀 Execution Guide
* **Dashboard:** `streamlit run app/main.py`
* **Ingestion:** `python -m src.rag.vector_store`
* **Evaluation:** `python scripts/evaluate_ragas.py`

---
*Developed for Canadian Financial Services Compliance. Air-Gapped. Secure. Semantic.*
