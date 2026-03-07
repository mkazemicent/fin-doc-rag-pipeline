# ADR 0007: Semantic-Aware Recursive Chunking Strategy

**Status:** Accepted
**Date:** March 2026

## Context
While ADR 0007 initially proposed `SemanticChunker`, production testing (UAT) revealed that pure embedding-based splitting is computationally expensive and operationally fragile. Specifically, it frequently exceeded the context window of local Ollama models (512 tokens), leading to `400 Bad Request` errors and inconsistent metadata preservation during the split.

## Decision
We have pivoted to a **Semantic-Aware Recursive Strategy** for v1.0:

1. **Mechanism:** Utilizes `RecursiveCharacterTextSplitter` with specialized separators: `["\n\n", "\n", ". ", " ", ""]`.
2. **Parameters:** 
    * `chunk_size = 800` characters (ensuring 100% compatibility with local 512-token context windows).
    * `chunk_overlap = 150` characters (maintaining semantic continuity).
3. **Metadata Integrity:** Implemented a single-pass ingestion logic that injects `access_group="general"` and `source` metadata into every chunk at the moment of split.
4. **Governance Integration:** Tied directly to the SQLite `IngestionTracker` to ensure incremental ingestion of new or modified documents.

## Consequences

### Positive
* **100% Reliability:** Eliminated `400 Bad Request` errors related to chunk size.
* **Performance:** Significantly faster ingestion compared to NLP-based semantic splitting.
* **Deterministic Results:** The splitter predictably breaks at paragraphs and sentences before defaulting to space-based splits.

### Negative
* **Semantic Nuance:** Less "dynamic" than sentence-similarity splitting, though mitigated by the optimized separator list.
