.PHONY: up down db-init test test-int test-model test-llm test-all lint fmt \
	draft-questions eval-validate eval-retrieval eval

up:
	docker compose up -d db

down:
	docker compose down

db-init:
	uv run ttd db init

# unit tests only: no database, no model download
test:
	uv run pytest

# needs the Postgres container (make up)
test-int:
	uv run pytest -m integration

# runs the real embedding model; about 2.2 GB download on first run
test-model:
	uv run pytest -m model

# calls your real LLM server (LLM_* in .env)
test-llm:
	uv run pytest -m llm

test-all:
	uv run pytest -m ""

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff check --fix .
	uv run ruff format .

# --- evaluation (Day 3) ---

# draft candidate questions with the LLM (you must review them by hand)
draft-questions:
	uv run python -m eval.draft_questions

# check eval/questions.jsonl against the database and the corpus files
eval-validate:
	uv run python -m eval.validate_dataset

# retrieval metrics for all configurations, no LLM needed
eval-retrieval:
	uv run python -m eval.run_eval --no-generate

# everything: retrieval plus generated answers and judging
eval:
	uv run python -m eval.run_eval --generate
