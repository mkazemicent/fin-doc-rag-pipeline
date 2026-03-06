# ADR 0001: Local Vector Storage and PII Masking for Data Residency

**Status:** Accepted
**Date:** March 2026

## Context
As a Canadian financial institution, processing deal analytics (Syndicated Loans, M&A Term Sheets) involves handling highly sensitive Non-Public Information (NPI) and Personally Identifiable Information (PII). Sending unmasked corporate financial data to third-party cloud vector databases (e.g., Pinecone, Weaviate Cloud) or public LLM APIs violates strict OSFI data residency and privacy regulations. We need a secure pipeline to ingest documents, extract terms, and perform RAG without leaking sensitive data.

## Decision
1. **PII/NPI Masking Layer:** We are implementing **Microsoft Presidio** as a mandatory preprocessing step to redact NPI/PII before chunking.
2. **Local Vector Storage & Embeddings:** To guarantee zero data leakage, we are utilizing **Ollama** running locally on an NVIDIA RTX 3070 Ti. We have selected `mxbai-embed-large` (a state-of-the-art open-source embedding model for RAG) to generate embeddings, which are then persisted entirely locally using **ChromaDB**.

## Consequences

**Positive:**
* **Regulatory Compliance:** Guarantees that no raw, unmasked financial data leaves the secure environment.
* **Cost Efficiency:** Running ChromaDB locally incurs zero cloud storage costs during development and testing.
* **Security:** Eliminates the attack surface associated with managing API keys for external vector databases.

**Negative / Trade-offs:**


* **Scalability:** Local ChromaDB is excellent for single-node document processing but will require migration to an enterprise-grade, self-hosted vector database (like self-hosted Milvus or pgvector) if the application scales to process thousands of concurrent deals.
* **Latency:** Adding the Presidio NLP masking layer introduces a slight processing delay during the initial document ingestion phase.