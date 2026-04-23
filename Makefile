.PHONY:

install:
	uv sync

install-all:
	uv sync --all-extras

install-pipeline:
	uv sync --extra pipeline

zenml:
	uv run zenml connect --url http://localhost:8080 --username admin --password Admin@123


data:
	uv run python -m pipeline.cli --run-prepare-data --no-cache

generation:
	uv run python -m pipeline.cli --run-generate-gmbl --no-cache

generate_question:
	uv run python -m pipeline.cli --run-generate-questions --no-cache

upload:
	uv run python -m pipeline.cli --run-upload-dataset

simple_finetune:
	uv run python -m pipeline.cli --run-finetune

option_finetune:
	uv run python -m pipeline.cli --run-finetune --num-epochs 1 --batch-size 2 --learning-rate 2e-4

simple_finetune_peft_acemath:
	uv run python -m pipeline.cli --run-finetune-peft-acemath

smoke_finetune_peft_acemath:
	uv run python -m pipeline.cli --run-finetune-peft-acemath --num-epochs 1 --batch-size 1 --learning-rate 2e-4 --dataset-workspace quangne --dataset-repo geometry3k8-8-1-1 --is-dummy --dummy-train-samples 20 --dummy-eval-samples 5

full_finetune_peft_acemath:
	uv run python -m pipeline.cli --run-finetune-peft-acemath --num-epochs 6 --batch-size 2 --learning-rate 2e-4 --dataset-workspace quangne --dataset-repo geometry3k8-8-1-1


aws_sagemaker_roles:
	uv run python -m src.infrastructures.aws.roles.create_sagemaker_role


aws_excecution_roles:
	uv run python -m src.infrastructures.aws.roles.create_execution_role


deploy_endpoint:
	uv run python -m src.infrastructures.aws.deploy.huggingface.run

del_endpoint:
	uv run python -m src.infrastructures.aws.deploy.delete_sagemaker_endpoint

del_endpoint_config:
	uv run python -m src.infrastructures.aws.deploy.delete_sagemaker_endpoint_config

endpoint:
	uv run python -m src.main

llm_local:
	uv run python scripts/run_vllm_local.py

llm_local_bg:
	uv run python scripts/run_vllm_local.py --detach

llm_local_health:
	curl http://localhost:8001/v1/models

worker:
	uv run python -m celery -A src.infrastructures.celery.config worker --loglevel=info --concurrency=1

flower:
	uv sync --group monitoring
	uv run python -m flower --app=src.infrastructures.celery.config --port=5555

rabbitmq_status:
	docker exec rabbitmq rabbitmqctl list_queues


docker_up:
	docker compose up -d

docker_infra:
	docker compose up -d postgres rabbitmq redis

migrate:
	uv run alembic upgrade head

revision:
	uv run alembic revision --autogenerate -m "upgrade tables"

mock:
	curl -X POST http://vllm:8000/v1/completions \
	-H "Content-Type: application/json" \
	-d '{"prompt":"Chuyển bài toán hình học tiếng Việt sang Geometry DSL (S-expression).\nChỉ trả về DSL thuần văn bản hợp lệ từ đề bài, không markdown, không giải thích.\nBỏ qua phần yêu cầu chứng minh hoặc câu hỏi phụ, nhưng giữ mọi dữ kiện hình học và điều kiện ràng buộc trong đề.\n\nĐề bài:\nCho tam giác ABC vuông tại A, có góc B bằng 30 độ\n\nDSL:"}'

compose_vllm:
	docker compose -f compose.vllm.yaml up -d

compose_infra:
	docker compose -f compose.yaml up -d

compose_infra_vol_down:
	docker compose -f compose.yaml down -v

compose_app:
	docker compose -f compose.yaml -f compose.app.yaml up -d