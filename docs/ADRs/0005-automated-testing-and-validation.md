# ADR 0005: Comprehensive Automated Testing and Validation Strategy

**Status:** Accepted
**Date:** March 2026

## Context
While ADR 0003 established RAGAS for evaluating the probabilistic output of our LLM (Faithfulness and Context Precision), the Enterprise Financial Deal Analyzer currently lacks a deterministic testing framework. To ensure data privacy compliance (verifying Microsoft Presidio actually masks PII), structural integrity (preventing chunk size overflows), and resilient agent routing, we must implement a rigorous testing ecosystem that covers all components of the pipeline prior to deployment.

## Decision
We are adopting a multi-layered testing architecture spanning the entire application lifecycle:

1. **Unit Testing Framework (`pytest`):** We will use `pytest` as our core framework to validate all deterministic functions across every module.
    * **Ingestion Layer:** Asserting that the `PIIMasker` accurately identifies and redacts synthetic PII from raw text strings.
    * **Vector Layer:** Validating that the `RecursiveCharacterTextSplitter` strictly adheres to the 500-character limit and 50-character overlap configuration.
    * **Agent Layer:** Testing LangGraph state transitions, ensuring fallback mechanisms (e.g., returning the original prompt if the query optimizer fails) execute correctly.
2. **Integration Testing:** We will implement automated tests to verify the handoffs between systems, such as ensuring documents embedded in the local ChromaDB can be successfully retrieved by a mock query.
3. **Continuous Integration (CI) Guardrails:** In future phases, these test suites will run automatically on every code commit. If the privacy unit tests fail, the build will break, preventing unmasked data processing logic from reaching production.

## Consequences

**Positive:**
* **Compliance Assurance:** Mathematically proves to risk and compliance teams that the PII masking layer functions as designed before touching real financial data.
* **Regression Prevention:** Ensures future updates to the pipeline (like switching embedding models or altering chunk sizes) do not silently break existing functionality.
* **System Stability:** Validates error handling and batch processing limits, ensuring the local hardware (RTX 3070 Ti) is not overwhelmed.

**Negative / Trade-offs:**
* **Development Overhead:** Writing and maintaining comprehensive unit tests and mock objects requires significant upfront engineering time, slowing immediate feature development.