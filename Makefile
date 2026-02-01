.PHONY:

uv:
	uv sync --all-groups

zenml:
	uv run zenml connect --url http://localhost:8080 --username admin --password Admin@123

zenml_status:
	uv run zenml status

data:
	PYTHONPATH=. uv run python -m tools.run --run-prepare-data --no-cache

generation:
	PYTHONPATH=. uv run python tools/run.py --run-generate-gmbl --no-cache

upload:
	PYTHONPATH=. uv run python -m tools.run --run-upload-dataset

simple_finetune:
	PYTHONPATH=. uv run python tools/run.py --run-finetune

option_finetune:
	PYTHONPATH=. uv run python tools/run.py --run-finetune --num-epochs 1 --batch-size 2 --learning-rate 2e-4

aws_excecution_roles:
	PYTHONPATH=. uv run python src/infrastructures/aws/roles/create_execution_role.py

aws_sagemaker_roles:
	PYTHONPATH=. uv run python src/infrastructures/aws/roles/create_sagemaker_role.py

deploy_endpoint:
	PYTHONPATH=. uv run python src/infrastructures/aws/deploy/huggingface/run.py

del_endpoint:
	PYTHONPATH=. uv run python src/infrastructures/aws/deploy/delete_sagemaker_endpoint.py

del_endpoint_config:
	PYTHONPATH=. uv run python src/infrastructures/aws/deploy/delete_sagemaker_endpoint_config.py