from zenml import step
from huggingface_hub import HfApi, create_repo
from pathlib import Path
from loguru import logger
import os


@step
def push_to_hub_step(
    model_path: str,
    repo_name: str,
    hf_token: str,
    private: bool = False
) -> str:
    """Push merged model to HuggingFace Hub"""

    logger.info(f"Pushing model to HuggingFace Hub: {repo_name}")

    api = HfApi(token=hf_token)

    try:
        repo_url = create_repo(
            repo_name,
            token=hf_token,
            private=private,
            exist_ok=True
        )
        logger.info(f"Repository created/exists: {repo_url}")
    except Exception as e:
        logger.warning(f"Repo might exist: {e}")

    logger.info(f"Uploading files from {model_path}")
    api.upload_folder(
        folder_path=model_path,
        repo_id=repo_name,
        token=hf_token,
        commit_message="Upload fine-tuned GeoUni model for line segments"
    )

    hub_url = f"https://huggingface.co/{repo_name}"
    logger.success(f"Model pushed to {hub_url}")

    return hub_url
