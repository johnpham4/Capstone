from pathlib import Path
import re

import boto3
from loguru import logger
from sagemaker.huggingface import HuggingFace
from sagemaker.session import Session

from src.config.settings.settings import settings

finetuning_dir = Path(__file__).resolve().parent
finetuning_requirements_path = finetuning_dir / "requirements.txt"


def _build_comet_experiment_name(model_name: str, dataset_repo_name: str) -> str:
    model_part = model_name.strip().split("/")[-1]
    model_part = model_part.replace(" ", "").replace("_", "")
    model_part = re.sub(r"(?i)(-?instruct)$", "", model_part)
    model_part = re.sub(r"[^0-9A-Za-z.-]", "", model_part)

    dataset_part = dataset_repo_name.strip().split("/")[-1]
    dataset_part = dataset_part.replace(" ", "")
    dataset_part = re.sub(r"[^0-9A-Za-z.-]", "", dataset_part)

    if not model_part:
        model_part = "unknown"
    if not dataset_part:
        dataset_part = "unknown"

    return f"{model_part}_{dataset_part}"


def run_finetuning_on_sagemaker(
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    dataset_huggingface_workspace: str = "quangne",
    dataset_huggingface_repo_name: str = "geometry3k8-8-1-1",
    model_output_huggingface_workspace: str = "quangne",
    model_name: str = "nvidia/AceMath-1.5B-Instruct",
    is_dummy: bool = False,
    dummy_train_samples: int = 400,
    dummy_eval_samples: int = 100,
    instance_type: str = "ml.g5.xlarge",
) -> None:
    assert settings.HF_TOKEN, "Hugging Face access token is required. Set HF_TOKEN in .env"
    assert settings.AWS_ARN_ROLE, "AWS ARN role is required. Set AWS_ARN_ROLE in .env"
    assert settings.AWS_ACCESS_KEY_ID, "AWS_ACCESS_KEY_ID is required. Set it in .env"
    assert settings.AWS_SECRET_ACCESS_KEY, "AWS_SECRET_ACCESS_KEY is required. Set it in .env"
    assert settings.AWS_REGION, "AWS_REGION is required. Set it in .env"

    if not finetuning_dir.exists():
        raise FileNotFoundError(f"The directory {finetuning_dir} does not exist.")
    if not finetuning_requirements_path.exists():
        raise FileNotFoundError(f"The file {finetuning_requirements_path} does not exist.")

    boto_session = boto3.Session(
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        region_name=settings.AWS_REGION,
    )
    sagemaker_session = Session(boto_session=boto_session)

    logger.info(f"Model output Hugging Face workspace: {model_output_huggingface_workspace}")
    comet_experiment_name = _build_comet_experiment_name(
        model_name=model_name,
        dataset_repo_name=dataset_huggingface_repo_name,
    )
    logger.info(f"Comet experiment name: {comet_experiment_name}")

    hyperparameters = {
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "dataset_huggingface_workspace": dataset_huggingface_workspace,
        "dataset_huggingface_repo_name": dataset_huggingface_repo_name,
        "model_name": model_name,
        "model_output_huggingface_workspace": model_output_huggingface_workspace,
    }
    if is_dummy:
        hyperparameters["is_dummy"] = True
        hyperparameters["dummy_train_samples"] = dummy_train_samples
        hyperparameters["dummy_eval_samples"] = dummy_eval_samples
        logger.info(
            "Dummy mode enabled for PEFT training: "
            f"train_samples={dummy_train_samples}, eval_samples={dummy_eval_samples}"
        )

    # Create the HuggingFace SageMaker estimator
    huggingface_estimator = HuggingFace(
        entry_point="finetune.py",
        source_dir=str(finetuning_dir),
        instance_type=instance_type,
        instance_count=1,
        role=settings.AWS_ARN_ROLE,
        sagemaker_session=sagemaker_session,
        py_version="py311",
        image_uri="763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:2.5.1-transformers4.49.0-gpu-py311-cu124-ubuntu22.04-v2.3",
        hyperparameters=hyperparameters,
        requirements_file=finetuning_requirements_path,
        environment={
            "HUGGING_FACE_HUB_TOKEN": settings.HF_TOKEN,
            "COMET_API_KEY": settings.COMET_API_KEY,
            "COMET_PROJECT_NAME": settings.COMET_PROJECT,
            "COMET_EXPERIMENT_NAME": comet_experiment_name,
        },
    )

    # Start the training job on SageMaker.
    huggingface_estimator.fit()


if __name__ == "__main__":
    run_finetuning_on_sagemaker()
