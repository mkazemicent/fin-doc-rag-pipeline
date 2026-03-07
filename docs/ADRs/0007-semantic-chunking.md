# ADR 0007: Transition to Semantic Chunking

## Status
Accepted

## Context
Our initial implementation used `RecursiveCharacterTextSplitter` with fixed chunk sizes (500 characters) and overlaps (50 characters). While computationally efficient, this approach often splits financial clauses mid-sentence or mid-paragraph, which can lead to:
1. Loss of critical context for the LLM.
2. Poor retrieval relevancy when terms are separated across chunks.
3. Inconsistent extraction of structured data like maturity dates and interest rates.

## Decision
We will migrate the ingestion pipeline to use **Semantic Chunking** via the `langchain_experimental` package. 

### Implementation Details:
*   **Chunker:** `SemanticChunker`
*   **Embeddings:** Local `OllamaEmbeddings` (mxbai-embed-large).
*   **Breakpoint Type:** `percentile`. This threshold type analyzes the cosine distance between subsequent sentence embeddings and splits when the distance exceeds a certain percentile of all distances in the document.
*   **Deprecation Fix:** Updated `OllamaEmbeddings` import from `langchain_community` to `langchain_ollama`.

## Consequences

### Positive
*   **Context Integrity:** Chunks will now align with semantic shifts (e.g., between different contract clauses), providing more coherent context to the Agent.
*   **Improved RAGAS Scores:** We expect an increase in `faithfulness` and `answer_relevancy` scores as retrieved context is naturally more complete.

### Negative
*   **Compute Cost:** Generating embeddings for every sentence to determine split points increases the total time required for document ingestion.
*   **Variable Chunk Size:** Unlike fixed character splitting, semantic chunks can vary significantly in length, requiring monitorring of our local LLM's context window.

## Alternatives Considered
*   **Agentic Chunking:** Using an LLM to identify split points. Rejected due to excessive compute requirements for large document sets.
*   **Larger Fixed Chunks:** (e.g., 2000 characters). Rejected because it introduces more noise per chunk.