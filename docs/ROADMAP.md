# Project Roadmap & Future Enhancements

This document outlines the planned upgrades for the Enterprise Deal Analyzer pipeline to evolve it from a local PoC to a compliance-ready, enterprise-grade AI architecture.

## Phase 1: Core Pipeline (Completed)
- [x] Zero-trust document ingestion with Microsoft Presidio PII masking.
- [x] 100% Air-gapped Vector Storage using local ChromaDB and `mxbai-embed-large`.
- [x] LangGraph State Machine implementation for agentic reasoning.
- [x] Autonomous Query Rewriting node to fix keyword imbalance and improve vector search.

## Phase 2: Automated Testing & Validation (In Progress)
- [x] **Unit Testing (Privacy):** Implement `pytest` to mathematically prove Presidio PII masking works against synthetic data.
- [x] **Unit Testing (Plumbing):** Build integration tests for `vector_store.py` to verify chunk size limits.
- [ ] **RAGAS Evaluation Integration:** Automate the scoring of Context Precision and Faithfulness to mathematically prove pipeline accuracy (ADR 0003).

## Phase 3: Enterprise Data Governance (Upcoming)
- [x] **Incremental Ingestion:** Implement file hashing (SQLite/JSON) to allow dynamic, on-the-fly document uploads without reprocessing the entire dataset.
- [x] **Document Lifecycle & Deletion (TTL):** Tag all ChromaDB chunks with unique document IDs to allow targeted deletion of specific contracts.
- [ ] **Role-Based Access Control (RBAC):** Draft the security model for isolating deal access between different analyst teams.

## Phase 4: Advanced Retrieval & Dynamic Routing (Upcoming)
- [ ] **Agentic / Semantic Chunking:** Replace `RecursiveCharacterTextSplitter` with an NLP model to dynamically split chunks based on topic/meaning.
- [ ] **Hybrid Search:** Implement BM25 (keyword search) alongside Vector (semantic) search to guarantee exact document ID retrieval.
- [x] **Dynamic LangGraph Routing:** Add a "Grade Context" self-correction loop to retry searches if the initial retrieval fails.

## Phase 5: Enterprise UI & Deployment (Upcoming)
- [ ] Build a frontend Streamlit dashboard featuring system diagnostics and status indicators.
- [ ] Implement "Rock-Solid Citations" with expandable context panels showing exact source documents.
- [ ] Add an "Export to Memo" feature for analysts to download generated insights.
- [ ] Containerize the application using Docker for secure, isolated deployment.