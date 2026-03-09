# ADR 0009: Centralized Configuration and Dependency Injection

**Status:** Accepted
**Date:** March 2026

## Context

The codebase had 20+ scattered `os.getenv()` calls across 5 files (`app/main.py`, `deal_analyzer.py`, `chroma_deal_store.py`, `document_processor.py`, `evaluate_ragas.py`), each with independent default values. This created several problems:

1. **Configuration drift:** Defaults were duplicated and could diverge silently.
2. **Untestable code:** Tests had to use fragile `patch.dict(os.environ, {...})` or patch module-level constants, both of which break when import order changes.
3. **No validation:** Environment variables were parsed as raw strings with no type checking.
4. **Hidden dependencies:** LLM, retriever, and embeddings were instantiated inside node functions, making it impossible to share instances or inject test doubles.

## Decision

### 1. Pydantic BaseSettings (`src/config.py`)

All configuration is centralized in a single `Settings` class using `pydantic_settings.BaseSettings`:

```python
class Settings(BaseSettings):
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3.1"
    embedding_model: str = "mxbai-embed-large"
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    # ... 15+ typed fields with defaults
    model_config = SettingsConfigDict(env_file=".env.local", extra="ignore")
```

A `@lru_cache` singleton (`get_settings()`) provides the default instance. All modules import from `src.config` instead of calling `os.getenv()`.

### 2. Dependency Injection via `functools.partial`

LangGraph node functions are module-level (for direct test imports) with optional keyword-only dependencies:

```python
def rewrite_node(state: AgentState, *, llm=None):
    if llm is None:
        llm = ChatOllama(...)  # fallback for standalone calls
    ...
```

`build_deal_analyzer()` creates shared instances once and binds them via `partial`:

```python
def build_deal_analyzer(settings=None):
    llm = ChatOllama(...)       # created ONCE
    retriever = ChromaDealStore(settings).get_retriever()
    workflow.add_node("rewrite", partial(rewrite_node, llm=llm))
```

### 3. Settings Injection in Tests

Tests construct `Settings(data_dir=tmp_path, llm_model="test-llm")` directly and pass it to constructors. No `os.environ` patching required.

## Consequences

### Positive
* **Single source of truth:** All defaults and types in one file.
* **Type-safe:** Pydantic validates `int`, `float`, `Path`, `str` at startup.
* **Testable:** `Settings(...)` constructor accepts overrides directly. Tests are deterministic and decoupled from environment.
* **Shared instances:** LLM, retriever, and reranker created once per agent build, not per node invocation.

### Negative
* **Import dependency:** All modules now depend on `src.config`. Circular imports must be avoided (resolved by lazy imports in `build_deal_analyzer`).
* **Cache invalidation:** `@lru_cache` on `get_settings()` means environment changes at runtime require `get_settings.cache_clear()`.
