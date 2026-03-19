# ============================================================
# Enterprise Deal Analyzer — Workflow Commands
#
# MODES (never run more than one simultaneously):
#   make test     → unit tests only, no services needed
#   make services → ChromaDB + Ollama in Docker (local dev)
#   make dev      → services + Streamlit in venv
#   make demo     → full Docker stack (app + services)
#   make down     → stop all Docker services
# ============================================================

.PHONY: test services dev demo down reset logs

## Run unit tests (venv only, no Docker services needed)
test:
	pytest tests/ -v

## Run tests with coverage report
test-cov:
	pytest tests/ -v --cov=src --cov-report=term-missing

## Start only ChromaDB + Ollama (use for local dev with venv Streamlit)
services:
	docker compose up -d chromadb ollama

## Start services + run Streamlit locally (local dev mode)
## WARNING: Do not run this if 'make demo' is already running
dev: services
	streamlit run app/main.py

## Start full Docker stack — app + chromadb + ollama
## WARNING: Do not run 'make dev' while this is running
demo:
	docker compose up -d --build

## Stop all Docker services
down:
	docker compose down

## Reset ChromaDB collection and ingestion tracker (required after model/chunk changes)
reset:
	python -c "from src.rag.chroma_deal_store import ChromaDealStore; s = ChromaDealStore(); s.reset_collection(); print('Collection and tracker reset.')"

## Tail app logs (Docker demo mode)
logs:
	docker logs deal-analyzer-app -f

## Rebuild Docker image without cache (use after dependency changes)
rebuild:
	docker compose build --no-cache app
