# Project Roadmap: v1.0, v2.0 & Future Enhancements

This document outlines the architectural journey of the Enterprise Deal Analyzer and the planned upgrades for future iterations.

## v1.0 Release (Completed)
- [x] **Zero-Trust Ingestion:** Microsoft Presidio PII masking integrated into the ingestion pipeline.
- [x] **Air-Gapped Vector DB:** Local ChromaDB deployment using `mxbai-embed-large`.
- [x] **Agentic Reasoning:** LangGraph-based state machine with autonomous query rewriting.
- [x] **Data Governance:** SQLite-based SHA-256 hash tracking for incremental ingestion and deduplication.
- [x] **Validation Framework:** 71% statement coverage with standardized `pytest` and `pyproject.toml`.
- [x] **Enterprise UI:** Streamlit dashboard with system diagnostics and retrieval transparency cards.
- [x] **Semantic-Aware Recursive Strategy:** Optimized chunking (800/150) for maximum reliability and local LLM compatibility.

## v2.0 Release (Completed)

### Retrieval Quality
- [x] **Two-Stage Semantic Chunking:** `SemanticChunker` (embedding-based boundary detection) with `RecursiveCharacterTextSplitter` size-cap fallback. Preserves financial section boundaries. ([ADR 0007](ADRs/0007-semantic-chunking.md))
- [x] **Cross-Encoder Reranking:** FlashRank `ms-marco-MiniLM-L-12-v2` (CPU-only, 22MB) reranks 20 MMR candidates to top 5 by query-document relevance. ([ADR 0010](ADRs/0010-retrieval-quality-pipeline.md))
- [x] **MMR Diversity Search:** Maximum Marginal Relevance (`fetch_k=80`, `k=20`, `lambda_mult=0.7`) replaces pure similarity search, eliminating redundant chunks.

### Architecture & Code Quality
- [x] **Centralized Configuration:** Pydantic `BaseSettings` replaces 20+ scattered `os.getenv()` calls. Single source of truth with typed validation. ([ADR 0009](ADRs/0009-centralized-configuration.md))
- [x] **Dependency Injection:** `functools.partial` binds shared LLM/retriever/reranker instances into LangGraph nodes. LLM created once per agent build, not per query.
- [x] **Typed Routing Signals:** `QueryStatus(str, Enum)` with `RELEVANT`, `IRRELEVANT`, `ERROR` values replaces fragile string matching for state machine routing.

### UI Bug Fixes & Security
- [x] **Transparency Gating:** `routing_signal == "relevant"` gates retrieval transparency display, fixing the bug where refusal answers showed documents.
- [x] **DealExtraction Serialization:** Proper human-readable serialization in chat history instead of raw Pydantic `repr()`.
- [x] **XSS Prevention:** `html.escape()` on all user-controlled content rendered with `unsafe_allow_html=True`.
- [x] **History Routing Signal:** `routing_signal` stored in message data and checked during page-reload re-rendering.

### Testing
- [x] **57 Tests, 9 Test Files:** Expanded from 28 tests with new coverage for semantic chunking, retrieval quality (MMR + reranker), transparency logic, XSS escaping, and DealExtraction serialization.

## Next Generation: v3.0 (Planned)

### Retrieval Enhancements
- [ ] **Hybrid Search:** BM25 keyword search alongside vector search for exact financial term matching (e.g., "SOFR", "2.50%").
- [ ] **Metadata Pre-filtering:** Hard filters at retrieval stage based on `access_group` or date ranges.
- [ ] **Document Summarization Indexing:** Secondary index of document summaries for portfolio-level queries and cross-deal trend analysis.

### Deployment & DevOps
- [ ] **Observability:** OpenTelemetry tracing for agent node latency, retrieval hit rates, and reranker effectiveness.
- [ ] **CI/CD Pipeline:** GitHub Actions with automated `pytest`, `ruff`, and coverage gates.
- [ ] **Helm Chart:** Kubernetes-ready deployment for Red Hat OpenShift.

---
*Enterprise RAG v2.0 | March 2026*
