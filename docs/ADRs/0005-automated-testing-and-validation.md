# ADR 0005: Standardized Automated Testing and Validation Architecture

**Status:** Accepted
**Date:** March 2026

## Context
Initial attempts at testing relied on fragile `sys.path` hacks within individual test files. This led to collection errors when running `pytest` from the project root and inconsistent import behavior between modules. Furthermore, we needed a strategy to validate complex agent routing and metadata preservation without dependency on a live GPU or local LLM.

## Decision
We have standardized the testing architecture by implementing a modern Python package structure:

1. **`pyproject.toml` Configuration:** Established as the central configuration for `pytest`. It explicitly sets `pythonpath = ["."]` and defines the test directory, eliminating the need for `sys.path` injection in code.
2. **Package Designation:** Added `src/__init__.py` to officially designate the `src` directory as a Python package, ensuring consistent absolute imports (e.g., `from src.rag...`) across tests and production code.
3. **Mock-First Methodology:** Mandated the use of `unittest.mock` and `patch` for all LLM and Embedding calls. This ensures the test suite is deterministic, fast, and can run on CI/CD environments without specialized hardware.
4. **Coverage Guardrails:** Established a requirement for statement coverage (currently at 71%) to ensure critical path logic (routing, hash checking, metadata injection) is validated.

## Consequences

### Positive
* **Developer Productivity:** Standardized commands (`python -m pytest`) work immediately for all contributors.
* **Operational Reliability:** Guaranteed validation of governance features (hash tracking, PII masking) on every build.
* **Portability:** The testing framework is no longer tied to local file system quirks.

### Negative
* **Brevity:** Requires more structured test code and mock configurations compared to simple script-based testing.
