# ADR 0003: Automated Pipeline Evaluation using RAGAS

**Status:** Accepted
**Date:** March 2026

## Context
Generative AI models are prone to hallucinations. In the context of Deal Analytics (processing SEDAR+ credit agreements and ISDA Master Agreements), traditional deterministic unit testing is insufficient to validate the extraction of complex financial terms and risk assessments. We need a robust, quantitative framework to continuously evaluate the accuracy of both the retrieval mechanism (vector search) and the generation mechanism (LLM output) to ensure enterprise-grade reliability.

## Decision
We are integrating **RAGAS (Retrieval Augmented Generation Assessment)** as our primary evaluation framework. Specifically, we will track two critical metrics:
1. **Faithfulness:** Measures if the generated answer (e.g., extracted loan amount) is grounded entirely in the retrieved context, penalizing any LLM hallucinations.
2. **Context Precision:** Measures whether the retrieval step successfully fetched the most relevant chunks from the dense legal PDFs and ranked them correctly for the LLM.

## Consequences

**Positive:**
* **Objective Trust:** Provides quantitative metrics (0.0 to 1.0 scores) to prove to stakeholders that the pipeline is safe for compliance-heavy environments.
* **Iterative Improvement:** Allows us to mathematically measure if tweaking our chunk size or LangGraph prompt actually improves performance.

**Negative / Trade-offs:**
* **Manual Effort Required:** We must manually create a "Golden Dataset" of 10-20 ground-truth Question/Answer pairs based on our raw SEDAR+ documents to run the evaluations.
* **Compute Cost:** Running RAGAS requires making additional LLM API calls to evaluate the pipeline's outputs, marginally increasing development costs.