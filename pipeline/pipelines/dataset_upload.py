<<<<<<< HEAD:pipelines/dataset_upload.py
from zenml import pipeline
from loguru import logger

from steps.upload import upload_to_huggingface
from src.config.settings.base import settings

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

=======
from zenml import pipeline
from loguru import logger

from pipeline.steps.upload import upload_to_huggingface
from src.config.settings.base import settings

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

>>>>>>> 6cf03dda8dad8bb8fa1226b8b4e9166c3f287527:pipeline/pipelines/dataset_upload.py
