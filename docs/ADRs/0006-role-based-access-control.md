# ADR 0006: Role-Based Access Control (RBAC) Security Model

**Status:** Accepted
**Date:** March 2026

## Context
In a highly regulated Canadian banking environment, Information Barriers (often called "Ethical Walls") are legally mandated. An analyst in the M&A advisory group must not have access to Non-Public Information (NPI) belonging to the Corporate Lending group. If our Enterprise Deal Analyzer allows any user to query the entire ChromaDB vector space globally, it violates these compliance laws. We need a systemic method to restrict document retrieval based on the authenticated user's authorization level.

## Decision
We will implement Role-Based Access Control (RBAC) at the **database retrieval level**, rather than just hiding information in the UI. 

1. **Ingestion Metadata Tagging:** During the ingestion phase, every document will be assigned an `access_group` metadata tag (e.g., `access_group: "syndicated_loans"` or `access_group: "global_compliance"`).
2. **Dynamic Query Filtering:** When a user submits a question via the UI, the application will pass their authenticated role to the LangGraph agent. The retrieval node will dynamically inject a metadata filter into the ChromaDB search (e.g., `retriever.invoke(query, filter={"access_group": user_role})`).
3. **Default Deny:** The system will operate on a "default deny" principle. If a user has no assigned role, the vector search will automatically return zero chunks.

## Consequences

**Positive:**
* **Regulatory Compliance:** Mathematically guarantees data isolation at the vector database level, satisfying OSFI Information Barrier requirements.
* **Auditability:** Security teams can easily audit the ChromaDB metadata to verify access controls are properly mapped.

**Negative / Trade-offs:**
* **Ingestion Complexity:** Requires us to map every raw PDF to a specific security group before running the pipeline.
* **Authentication Overhead:** The Streamlit frontend will eventually require integration with an identity provider (like Azure Active Directory or Okta) to securely pass user roles to the backend agent.