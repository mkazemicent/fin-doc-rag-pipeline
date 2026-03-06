# Project Roadmap & Future Enhancements

This document outlines the planned upgrades for the Enterprise Deal Analyzer pipeline to evolve it from a local PoC to a compliance-ready, enterprise-grade AI architecture.

## Phase 1: Core Pipeline (Completed)
- [x] Zero-trust document ingestion with Microsoft Presidio PII masking.
- [x] 100% Air-gapped Vector Storage using local ChromaDB and `mxbai-embed-large`.
- [x] LangGraph State Machine implementation for agentic reasoning.
- [x] Autonomous Query Rewriting node to fix keyword imbalance and improve vector search.

## Phase 2: Enterprise Data Governance (New)
- [ ] **Incremental Ingestion:** Implement file hashing to allow dynamic, on-the-fly document uploads via the UI without reprocessing the entire dataset.
- [ ] **Document Lifecycle & Deletion (TTL):** Tag all ChromaDB chunks with unique document IDs to allow targeted deletion of specific contracts (Right to be Forgotten compliance).
- [ ] **Role-Based Access Control (RBAC) Architecture:** Draft the security model for isolating deal access between different analyst teams (Ethical Walls).

## Phase 3: Advanced Retrieval & Dynamic Routing (Upcoming)
- [ ] **Agentic / Semantic Chunking:** Replace `RecursiveCharacterTextSplitter` with an NLP model to dynamically split chunks based on topic/meaning, preserving complete legal clauses.
- [ ] **Hybrid Search:** Implement BM25 (keyword search) alongside Vector (semantic) search to guarantee exact document ID or numeric figure retrieval.
- [ ] **Dynamic LangGraph Routing:** Add a "Grade Context" self-correction loop to retry searches if the initial retrieval fails, and an "Intent Router" for non-search queries.
- [ ] **RAGAS Evaluation Integration:** Automate the scoring of Context Precision and Faithfulness to mathematically prove pipeline accuracy.

## Phase 4: Enterprise UI & Deployment
- [ ] Build a frontend Streamlit dashboard featuring system diagnostics and status indicators.
- [ ] Implement "Rock-Solid Citations" with expandable context panels showing exact source documents and page numbers.
- [ ] Add an "Export to Memo" feature for analysts to download generated insights.
- [ ] Containerize the application using Docker for secure, isolated deployment.