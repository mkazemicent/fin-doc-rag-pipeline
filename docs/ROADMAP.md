# Project Roadmap & Future Enhancements

This document outlines the planned upgrades for the Enterprise Deal Analyzer pipeline to further improve retrieval accuracy and reasoning capabilities.

## Phase 1: Core Pipeline (Completed)
- [x] Zero-trust document ingestion with Microsoft Presidio PII masking.
- [x] 100% Air-gapped Vector Storage using local ChromaDB and `mxbai-embed-large`.
- [x] LangGraph State Machine implementation for agentic reasoning.
- [x] Autonomous Query Rewriting node to fix keyword imbalance and improve vector search.

## Phase 2: Advanced Retrieval Strategies (Upcoming)
- [ ] **Agentic / Semantic Chunking:** Replace `RecursiveCharacterTextSplitter` with Semantic Chunking. Use an NLP model to dynamically split chunks based on the shift in topic/meaning (keeping entire legal clauses perfectly intact regardless of character count).
- [ ] **Hybrid Search:** Implement BM25 (keyword search) alongside our current Vector (semantic) search to ensure we never miss exact document IDs or specific numeric figures.
- [ ] **RAGAS Evaluation Integration:** Automate the scoring of Context Precision and Faithfulness using the RAGAS framework (as defined in ADR 0003).

## Phase 3: Deployment & UI
- [ ] Build a frontend Streamlit dashboard with citation/source-tracking UI.
- [ ] Containerize the application using Docker for secure cloud deployment.