# LLM Engineering Makefile

.PHONY: sync run-etl-maxime run-etl-paul lint format test clean

uv:
	uv sync --all-groups

zenml_up:
	docker compose -f compose.zenml.yml up -d

zenml_server:
	uv run zenml connect --url http://localhost:8080 --username admin --password Admin@123

zenml_status:
	uv run zenml status

etl:
	uv run python -m tools.run --run-legal-etl --no-cache

extract:
	uv run python -m tools.run --run-extract --no-cache

upload:
	uv run python -m tools.run --run-upload-dataset

fe:
	uv run python -m tools.run --run-feature-engineering --no-cache

test:
	uv run pytest tests/

test_unit:
	uv run pytest tests/integration/
