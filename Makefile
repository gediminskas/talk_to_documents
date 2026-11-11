.PHONY: up down db-init test test-int test-model test-all lint fmt

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

test-all:
	uv run pytest -m ""

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff check --fix .
	uv run ruff format .
