.PHONY:

install:
	uv sync

install-all:
	uv sync --all-extras

install-pipeline:
	uv sync --extra pipeline

zenml:
	uv run zenml connect --url http://localhost:8080 --username admin --password Admin@123

zenml_status:
	uv run zenml status

data:
	uv run python -m pipeline.cli --run-prepare-data --no-cache

generation:
	uv run python -m pipeline.cli --run-generate-gmbl --no-cache

generate_question:
	uv run python -m pipeline.cli --run-generate-questions --no-cache

render_diagram:
	PYTHONPATH=. uv run python tools/run.py --run-render-diagram --no-cache

upload:
	uv run python -m pipeline.cli --run-upload-dataset

simple_finetune:
	uv run python -m pipeline.cli --run-finetune

option_finetune:
	uv run python -m pipeline.cli --run-finetune --num-epochs 1 --batch-size 2 --learning-rate 2e-4


aws_sagemaker_roles:
	uv run python src/infrastructures/aws/roles/create_sagemaker_role.py


aws_excecution_roles:
	uv run python src/infrastructures/aws/roles/create_execution_role.py


deploy_endpoint:
	uv run python src/infrastructures/aws/deploy/huggingface/run.py

del_endpoint:
	uv run python src/infrastructures/aws/deploy/delete_sagemaker_endpoint.py

del_endpoint_config:
	uv run python src/infrastructures/aws/deploy/delete_sagemaker_endpoint_config.py

endpoint:
	uv run python -m src.main

worker:
	uv run python -m celery -A src.infrastructures.celery.config worker --loglevel=info --concurrency=4

flower:
	uv run python -m celery -A src.infrastructures.celery.config flower --port=5555

rabbitmq_status:
	docker exec rabbitmq rabbitmqctl list_queues

test_load:
	wrk -t3 -c3 -d60s --timeout 180s -s post.lua \
		http://localhost:8000/api/v1/diagrams/render

docker_up:
	docker compose up -d

docker_infra:
	docker compose up -d postgres rabbitmq redis

migrate:
	uv run alembic revision --autogenerate -m "upgrade tables" && \
	uv run alembic upgrade head


mock:
	curl -X POST https://victoria-communicable-sometimes.ngrok-free.dev/generate \
	-H "Content-Type: application/json" \
	-d '{"prompt":"Cho tam giác ABC vuông tại A, có góc B bằng 30 độ"}'


ui:
	uv run python -m ui.py