# LLM Engineering Makefile

.PHONY: sync run-etl-maxime run-etl-paul lint format test clean

uv:
	uv sync --all-groups

zenml_up:
	docker compose -f compose.yaml up -d

zenml:
	uv run zenml connect --url http://localhost:8080 --username admin --password Admin@123

zenml_status:
	uv run zenml status


ngrok_up:
	- curl -s http://localhost:4040/api/tunnels | grep -o 'https://[^"]*'

data:
	uv run python -m tools.run --run-prepare-data --no-cache

generation:
	uv run python tools/run.py --run-generate-gmbl --no-cache

upload:
	uv run python -m tools.run --run-upload-dataset

simple_finetune:
	uv run python tools/run.py --run-finetune

option_finetune:
	uv run python tools/run.py --run-finetune --num-epochs 1 --batch-size 2 --learning-rate 2e-4

aws_excecution_roles:
	uv run python llm_src/infrastructures/aws/roles/create_execution_role.py

aws_sagemaker_roles:
	uv run python llm_src/infrastructures/aws/roles/create_sagemaker_role.py

deploy_endpoint:
	uv run python llm_src/infrastructures/aws/deploy/huggingface/run.py

del_endpoint:
	uv run python llm_src/infrastructures/aws/deploy/delete_sagemaker_endpoint.py

del_endpoint_config:
	uv run python llm_src/infrastructures/aws/deploy/delete_sagemaker_endpoint_config.py