# ADR 0008: Local PoC to Enterprise Cloud & Private Cloud Scaling

## Status
Proposed

## Context
The current implementation of the Enterprise Deal Analyzer is a local, air-gapped Proof of Concept (PoC). While ideal for rapid prototyping, it lacks the high availability and horizontal scalability required for production. For Tier-1 Banks, the path to production often avoids public cloud providers (AWS/Azure) in favor of **On-Premise Private Clouds** or **Hybrid Cloud** environments.

## Decision Drivers
*   **Data Sovereignty:** Strict requirement to keep sensitive financial contracts within the bank's owned or controlled infrastructure.
*   **Regulatory Compliance:** Adherence to OSFI (Canada) or similar global mandates regarding third-party cloud concentration risk.
*   **Scalability:** Need for a container orchestration platform that supports GPU-accelerated workloads (NVIDIA A100/H100).
*   **Operational Consistency:** Aligning with existing bank infrastructure, typically **Red Hat OpenShift**.

## Proposed Scaling Architecture (The "Private Cloud" Path)

### 1. Orchestration: Red Hat OpenShift (OCP)
*   **Transition:** Move from Docker Compose to **Red Hat OpenShift Container Platform**.
*   **Rationale:** OpenShift provides an enterprise-grade Kubernetes distribution with enhanced security (SCCs), integrated CI/CD, and native support for GPU operators. This allows the bank to run the "App" and "ChromaDB" containers with the same orchestration logic as public cloud, but on-premise.

### 2. Private LLM Inference (On-Cluster)
*   **Transition:** Instead of public endpoints (Azure OpenAI), host LLMs internally within the OpenShift cluster.
*   **Technology:** Use **NVIDIA NIM**, **vLLM**, or **TGI (Text Generation Inference)** containers. 
*   **Benefit:** Zero data leakage. All prompt traffic and document context stays within the bank's internal network (East-West traffic).

### 3. Managed Vector Storage (Operator-Led)
*   **Transition:** Use **PostgreSQL with pgvector** managed by a Kubernetes Operator (e.g., CrunchyData or Zalando).
*   **Rationale:** This provides RDS-like features (backups, HA, scaling) while running on the bank's own storage arrays (SAN/NAS).

### 4. Storage & Ingestion
*   **Transition:** Move from local volumes to S3-compatible on-prem storage like **MinIO** or **Ceph**.
*   **Flow:** Document Upload -> MinIO Bucket -> Internal Webhook -> OpenShift Ingestion Pod.

## Consequences
*   **Positive:** Full control over the data lifecycle; no reliance on external CSPs (Cloud Service Providers); leverages existing bank security protocols.
*   **Negative:** Higher CAPEX/OPEX for managing internal GPU clusters and storage arrays.
*   **Portability:** The current PoC is **OpenShift-Ready**. It already uses a non-root `appuser` (UID 1000) and standard environment variables, ensuring it can be deployed to an OCP cluster with minimal manifest changes.

## Compliance & Governance (Bank Standards)
*   **Network:** No egress to the public internet; all dependencies (Python packages, Docker images) must be pulled from internal Artifactory/Quay mirrors.
*   **IAM:** Integration with internal Active Directory/LDAP via OpenShift OAuth.
*   **Encryption:** Mandatory encryption for all Persistent Volumes (PVs) using the bank's internal HSM (Hardware Security Module).
