# Enterprise Deal Analytics & Risk Extraction Pipeline

A production-ready, agentic RAG pipeline designed to automate the extraction of critical financial terms and assess risks in unstructured corporate contracts (e.g., Syndicated Loans, ISDA Agreements). 

Built with strict data privacy considerations for the Canadian financial services sector.

## The Business Problem
Processing complex, unstructured deal documents manually is time-consuming and prone to human error. Missing a conflicting maturity date or a poorly structured liability clause introduces significant institutional risk. 

## The Solution
This pipeline utilizes an Agentic Retrieval-Augmented Generation (RAG) architecture to ingest PDF contracts, completely mask Personally Identifiable Information (PII), and deploy an autonomous AI agent to extract terms, flag missing clauses, and generate stakeholder-ready summaries. 

## Core Tech Stack
* **Core Language:** Python 3.10+
* **LLM & Orchestration:** OpenAI API / LangChain / LangGraph
* **Vector Storage:** ChromaDB (Local deployment for data residency compliance)
* **Data Privacy:** Microsoft Presidio (PII/NPI anonymization)
* **Evaluation Framework:** RAGAS (Context Precision & Faithfulness tracking)
* **User Interface:** Streamlit

## Key Features
* **Compliance-First Ingestion:** All documents are passed through a zero-trust PII masking layer before any data is embedded or sent to an LLM.
* **Agentic Retrieval:** Utilizes LangGraph to give the AI specific "tools" (Term Extractor, Risk Assessor, Summarizer) rather than relying on standard, linear semantic search.
* **Automated Risk Red-Flagging:** Automatically cross-references extracted dates and financial figures to highlight term mismatches and missing standard clauses (e.g., Force Majeure).
