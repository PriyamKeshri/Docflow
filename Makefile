.PHONY: install test lint format up up-prod down logs n8n-import dataset eval

PY ?= .venv/bin/python

install:
	python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .

format:
	$(PY) -m ruff check --fix . && $(PY) -m ruff format .

up:
	docker compose up -d --build

up-prod:
	docker compose -f docker-compose.yml -f deploy/docker-compose.prod.yml up -d --build

down:
	docker compose down

logs:
	docker compose logs -f api watcher n8n

# Imports the workflows (inactive). Open http://localhost:5678, add credentials, activate.
n8n-import:
	docker compose exec n8n n8n import:workflow --separate --input=/workflows

# Regenerates the synthetic part of the eval set (deterministic).
dataset:
	$(PY) eval/generate_dataset.py

# Calls the real model API: costs money. Rewrites the results section of README.md.
eval:
	$(PY) -m docflow eval --update-readme README.md
