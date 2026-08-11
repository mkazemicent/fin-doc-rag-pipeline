# TODO — Road to a public demo

**Goal: a live, clickable URL by 2026-08-27.**

The pipeline works. `pytest tests/ -q` → **59 passed** on a clean checkout. What's missing is
not functionality — it's that the whole stack assumes a local GPU, so there is nothing anyone
can open.

This file is the working checklist. Tick boxes as you go and commit it with them ticked.

---

## The blocker, stated once

`docker-compose.yml` runs `qwen2.5:14b` on Ollama with `NUM_GPU=35` — 9+ GB of VRAM. No free
host provides that. Every task below exists to route around it.

**Ollama remains the default.** The air-gapped local-first design is the point of the project
and does not change. We are *adding* a hosted-inference path used only by the public demo.
The end state is a stronger claim than either mode alone:

> Runs fully air-gapped on-prem for regulated data — and here's a hosted demo you can click.

---

## Ground rules

- **Commit and push after each milestone.** Do not squash. The history is part of the artifact.
- **No refactors.** 59 tests pass. Every "while I'm in here" costs a day.
- **No new features.** Not MCP, not trajectory evals, not agent improvements. Those come after
  a URL exists, or not at all.
- **If a milestone stalls for more than half a day**, cut its scope, commit what works, move on.
  M4 matters more than M2 being elegant.
- **Ship ugly and on time.** An imperfect deployed thing beats a perfect undeployed one; the
  undeployed one is worth zero.

Baseline check before starting anything:

```bash
cd /home/sgidy/fin-doc-rag-pipeline
.venv/bin/python -m pytest tests/ -q      # expect: 59 passed
```

---

## M1 — Pluggable LLM + embedding provider

**Why:** five hard-coded constructors are the only thing tying the app to a local GPU.

Call sites, all reading the same `Settings` object from `src/config.py`:

| File | Approx. line | What |
|---|---|---|
| `src/rag/deal_analyzer.py` | 69 | `ChatOllama` — rewrite node |
| `src/rag/deal_analyzer.py` | 172 | `ChatOllama` — grade node |
| `src/rag/deal_analyzer.py` | 218 | `ChatOllama` — generate node |
| `src/rag/deal_analyzer.py` | 268 | `ChatOllama` — structured extraction |
| `src/rag/chroma_deal_store.py` | 27 | `OllamaEmbeddings` |

- [ ] Add `langchain-openai` to `pyproject.toml` dependencies
- [ ] Add to `Settings`: `llm_provider: str = "ollama"`, `embedding_provider: str = "ollama"`,
      `openai_api_key: str = ""`, `openai_llm_model: str = "gpt-4o-mini"`,
      `openai_embedding_model: str = "text-embedding-3-small"`
- [ ] New `src/rag/providers.py` with `get_llm(settings)` and `get_embeddings(settings)`
- [ ] Replace all five call sites with factory calls
- [ ] Raise a clear error on an unknown provider name — not a silent fallback
- [ ] Add `.env.example` entries for the new settings

**Done when:** `pytest tests/ -q` still reports **59 passed**, and with `LLM_PROVIDER=openai`
in `.env.local` a document analyses end to end.

> **Trap:** this milestone changes *where the model runs* and nothing else. Do not touch
> prompts, retrieval parameters, `num_ctx`, or the LangGraph edges. If tests go red, the
> change was too large — revert and split it.

---

## M2 — Embedded ChromaDB mode

**Why:** the demo host runs one process. A separate `chromadb` container is not available.

- [ ] Add `chroma_mode: str = "server"` to `Settings` (`server` | `embedded`)
- [ ] In `src/rag/chroma_deal_store.py`, construct a `PersistentClient` on a local path when
      `embedded`, keeping the existing `HttpClient` path for `server`
- [ ] Add `chroma_persist_dir` setting, default `./data/chroma_db`
- [ ] Verify ingestion and retrieval both work in embedded mode

**Done when:** the app starts and answers with no `chromadb` container running.

> **Trap:** `nomic-embed-text` and OpenAI `text-embedding-3-small` produce vectors of
> **different dimensions**. An index built with one is unusable by the other — the failure is
> a dimension-mismatch error at query time, not at ingest. Rebuild the demo index with
> whichever provider the demo actually uses.

---

## M3 — Demo corpus that loads on first paint

**Why:** a visitor currently lands on an empty upload box, has no document, and leaves.

- [ ] Add **one** synthetic or public-domain credit agreement (a real one with no licence is
      not worth the risk — generate a realistic synthetic one)
- [ ] Store it at `assets/demo/` — **not** under `data/`, which is gitignored; a demo file
      placed there will be silently untracked and the deploy will come up empty
- [ ] Ship a pre-built index for it, or ingest at startup if that is under ~10 seconds
- [ ] Show its analysis on first load, before any upload
- [ ] Keep the upload path working for visitors who bring their own

**Done when:** opening the URL in a private browser window shows extracted deal terms and risk
factors within a few seconds, with zero clicks.

---

## M4 — Deploy

Streamlit Community Cloud: free, deploys straight from this repo, handles secrets. Render or
Fly.io are equally fine. **Do not spend a day comparing them.**

- [ ] Pin the demo settings via host environment variables, not committed files
- [ ] `OPENAI_API_KEY` goes in the host's secret store — **never in the repo**
- [ ] **Set a hard spend cap on the OpenAI key.** A public demo is a public key spender;
      assume someone will find it and hammer it
- [ ] Use a dedicated key for the demo so it can be revoked without affecting anything else
- [ ] Cap upload size and page count so one PDF cannot burn the budget
- [ ] Confirm no secret is present in the deployed image or logs

**Done when:** the URL loads on a phone, on mobile data, in a private window.

---

## M5 — README that leads with the demo

The three mermaid diagrams are good and stay. They are simply not the *first* thing a reader
needs. Restructure the top of `README.md` to fit in one screen:

- [ ] **Live demo URL — first line, unmissable**
- [ ] One sentence on the problem, in the language of someone who works at a bank
- [ ] Three or four bullets on what it does
- [ ] One screenshot of a real analysis
- [ ] Short "run it locally, air-gapped" section — the differentiator, stated plainly
- [ ] Then the existing architecture diagrams, for a reader who scrolls

**Done when:** someone who has never seen the repo understands what it is in 30 seconds.

---

## M6 — Publish

- [ ] Set the repo's **About → Website** field to the live URL
- [ ] Add repo topics: `rag`, `langgraph`, `mlops`, `llm`, `kubernetes`, `terraform`
- [ ] Final push; confirm the commit history reads as real incremental work

---

## Progress log

Append one line per session — date, what changed, test count.

```
2026-08-07  baseline verified on clean checkout — 59 passed
```
