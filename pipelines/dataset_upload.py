from zenml import pipeline
from loguru import logger
from steps.upload import upload_to_huggingface


@pipeline
def dataset_upload_pipeline(
    dataset_path: str,
    repo_id: str,
    split: str = "train"
) -> int:
    """Upload dataset to HuggingFace Hub"""
    from llm_engineering.settings import settings

    num_uploaded = upload_to_huggingface(
        dataset_path=dataset_path,
        repo_id=repo_id,
        token=settings.HF_TOKEN,
        split=split
    )
    logger.info(f"Dataset upload completed: {num_uploaded} figures")
    return num_uploaded
