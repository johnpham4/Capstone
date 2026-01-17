from zenml import pipeline
from loguru import logger

from steps.upload import upload_to_huggingface
from llm_engineering.settings import settings

@pipeline
def dataset_upload_pipeline(
    dataset_dir: str,
    repo_id: str
):

    num_uploaded = upload_to_huggingface(
        dataset=dataset_dir,
        repo_id=repo_id,
        token=settings.HF_TOKEN
    )

    logger.info(f"Dataset upload completed: {num_uploaded} samples")
    return num_uploaded

