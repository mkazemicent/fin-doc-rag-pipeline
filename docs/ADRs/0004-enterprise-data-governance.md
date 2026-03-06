# ADR 0004: Enterprise Data Governance and Document Lifecycle Management

**Status:** Accepted
**Date:** March 2026

## Context
As the Enterprise Financial Deal Analyzer transitions from a static Proof-of-Concept to a dynamic application, we must address critical banking data governance requirements. Currently, our ingestion pipeline processes the entire dataset in a single batch, and our vector store lacks the granularity to delete specific documents. To meet institutional compliance standards (e.g., Right to be Forgotten, data retention limits, and Ethical Walls between analyst teams), the system requires dynamic ingestion, precise data lifecycle management, and foundational access controls.

## Decision
We are implementing a comprehensive Data Governance layer consisting of three core pillars:

1. **Incremental Ingestion via File Hashing:** We will introduce a lightweight tracking mechanism (e.g., a local SQLite database or JSON registry) to compute and store the cryptographic hash (SHA-256) of every ingested PDF. When new documents are uploaded via the UI, the system will check the hash, ensuring only net-new documents are passed through the Presidio PII masking and chunking pipeline.
2. **Targeted Document Deletion (Metadata Tagging):** We will mandate that every text chunk embedded into ChromaDB includes strict metadata tracking, specifically a unique `document_id` and `source_filename`. This allows the system to execute targeted deletion commands (e.g., `vectorstore.delete(where={"source": "amerigo_2015.pdf"})`) without requiring a full database rebuild.
3. **Foundational Role-Based Access Control (RBAC):** We will structure our retrieval metadata to support future Ethical Walls. Chunks will be tagged with access-level metadata, allowing the LangGraph retrieval node to dynamically filter semantic search results based on the querying analyst's permissions.

## Consequences

**Positive:**
* **Regulatory Compliance:** Enables targeted data purging, fulfilling legal requirements to destroy client deal data when mandates expire.
* **Operational Efficiency:** Incremental ingestion prevents the pipeline from freezing or wasting GPU compute on re-embedding existing documents when a single new contract is added.
* **Enterprise Scalability:** Sets the foundational architecture needed for multi-tenant or multi-team deployments within a financial institution.

**Negative / Trade-offs:**
* **Architectural Complexity:** Requires maintaining a secondary state-tracking database (SQLite/JSON) alongside ChromaDB to manage file hashes and metadata.
* **Ingestion Overhead:** Hashing files and cross-referencing the database adds a minor computational step prior to the Presidio NLP masking process.

graph TD
    A[User Uploads PDF via UI] --> B[Calculate SHA-256 File Hash]
    B --> C{Check SQLite Hash Tracker}
    
    C -- Hash Exists --> D[Skip Processing: Document Already in DB]
    
    C -- Hash is New --> E[Extract Text & Mask PII]
    E --> F[Chunk Document]
    F --> G[Generate Local Embeddings]
    G --> H[(Append to ChromaDB)]
    H --> I[(Save New Hash to SQLite)]
    
    I --> J[UI: Ingestion Complete]
    D --> J