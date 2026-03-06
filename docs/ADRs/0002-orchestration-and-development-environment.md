# ADR 0002: Orchestration Framework and Development Environment

**Status:** Accepted
**Date:** March 2026

## Context
The Deal Analytics Pipeline requires more than simple semantic search (naive RAG). To accurately assess risks, extract specific terms, and generate actionable summaries from complex SEDAR+ filings, the AI must be able to route tasks dynamically, use specialized tools, and self-correct if it cannot find a specific clause. We need a framework capable of cyclic, multi-step reasoning, as well as a development environment that supports testing agentic workflows.

## Decision
1. **Orchestration Framework:** We are adopting **LangGraph** (built on top of LangChain) to orchestrate the AI agents. Instead of a linear chain, LangGraph allows us to define the pipeline as a state machine where specialized nodes (e.g., "Term Extractor Agent," "Risk Assessor Agent") can loop and communicate until a verifiable output is achieved.
2. **Development Environment:** We are utilizing **Google Antigravity IDE** for development. Its native multi-agent orchestration and Agent Manager interface provide the optimal environment for building, observing, and debugging the complex, parallel agent behaviors required by this pipeline.

## Consequences

**Positive:**
* **Advanced Capability:** LangGraph supports the exact autonomous, multi-tool workflow required to parse dense Canadian corporate credit agreements.
* **Enterprise Alignment:** Demonstrates to financial institutions (like RBC) the ability to design state-of-the-art, graph-based AI architectures, moving beyond basic prompt engineering.

**Negative / Trade-offs:**
* **Complexity:** LangGraph introduces a steeper learning curve and more boilerplate code compared to standard LangChain sequential chains.
* **Debugging Overhead:** Tracking the exact path an agent took through a graph-based state machine requires more rigorous logging and evaluation.