from pathlib import Path

import boto3
from huggingface_hub import HfApi
from loguru import logger
from sagemaker.huggingface import HuggingFace
from sagemaker.session import Session

from src.config.settings.settings import settings

finetuning_dir = Path(__file__).resolve().parent
finetuning_requirements_path = finetuning_dir / "requirements.txt"


def run_finetuning_on_sagemaker(
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    dataset_huggingface_workspace: str = "minn4",
    dataset_huggingface_repo_name: str = "text2dsl",
    model_name: str = "nvidia/AceMath-1.5B-Instruct",
    is_dummy: bool = False,
    instance_type: str = "ml.g5.2xlarge",
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

    api = HfApi()
    user_info = api.whoami(token=settings.HF_TOKEN)
    huggingface_user = user_info["name"]
    logger.info(f"Current Hugging Face user: {huggingface_user}")

    hyperparameters = {
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "dataset_huggingface_workspace": dataset_huggingface_workspace,
        "dataset_huggingface_repo_name": dataset_huggingface_repo_name,
        "model_name": model_name,
        "model_output_huggingface_workspace": huggingface_user,
    }
    if is_dummy:
        hyperparameters["is_dummy"] = True

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
        },
    )

    # Start the training job on SageMaker.
    huggingface_estimator.fit()


if __name__ == "__main__":
    run_finetuning_on_sagemaker()
