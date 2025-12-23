# LLM Engineering Makefile

.PHONY: sync run-etl-maxime run-etl-paul lint format test clean

uv:
	uv sync --all-groups

zenml_up:
	docker compose -f compose.zenml.yml up -d

zenml:
	uv run zenml connect --url http://localhost:8080 --username admin --password Admin@123

zenml_status:
	uv run zenml status

extract:
	uv run python -m tools.run --run-extract --no-cache

upload:
	uv run python -m tools.run --run-upload-dataset

fe:
	uv run python -m tools.run --run-feature-engineering --no-cache

ngrok_up:
	- curl -s http://localhost:4040/api/tunnels | grep -o 'https://[^"]*'

encode:
	uv run python -m tools.run --encode-images

