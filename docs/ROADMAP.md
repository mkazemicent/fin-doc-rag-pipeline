# Project Roadmap: v1.0 & Future Enhancements

This document outlines the architectural journey of the Enterprise Deal Analyzer and the planned upgrades for future iterations.

## v1.0 Release (Completed)
- [x] **Zero-Trust Ingestion:** Microsoft Presidio PII masking integrated into the ingestion pipeline.
- [x] **Air-Gapped Vector DB:** Local ChromaDB deployment using `mxbai-embed-large`.
- [x] **Agentic Reasoning:** LangGraph-based state machine with autonomous query rewriting.
- [x] **Data Governance:** SQLite-based SHA-256 hash tracking for incremental ingestion and deduplication.
- [x] **Validation Framework:** 71% statement coverage with standardized `pytest` and `pyproject.toml`.
- [x] **Enterprise UI:** Streamlit dashboard with system diagnostics and retrieval transparency cards.
- [x] **Semantic-Aware Recursive Strategy:** Optimized chunking (800/150) for maximum reliability and local LLM compatibility.

## Next Generation: v2.0 (Planned)
### 1. Superior Retrieval Precision
- [ ] **Cross-Encoder Re-ranking:** Implement a local re-ranker (e.g., BGE-Reranker) to refine the Top-K results after initial semantic retrieval.
- [ ] **Hybrid Search:** Integrate BM25 keyword search alongside vector search to guarantee exact document ID and entity retrieval.

### 2. Scalable Data Management
- [ ] **Metadata Pre-filtering:** Implement hard filters at the retrieval stage based on `access_group` or date ranges to optimize performance.
- [ ] **Document Summarization Indexing:** Create a secondary index of document summaries to allow for high-level portfolio oversight and cross-deal trend analysis.

### 3. Deployment & DevOps
- [ ] **Containerization:** Finalize Docker configuration for secure, isolated, and reproducible deployments across bank infrastructure.
- [ ] **Observability:** Integrate local logging and tracing (e.g., LangSmith local or OpenTelemetry) to monitor agent performance in production.

---
*Enterprise RAG v1.0 Release Date: March 2026*
