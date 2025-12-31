from zenml import pipeline
from loguru import logger

from steps.upload.upload_gmbl_dataset import upload_gmbl_to_huggingface


@pipeline
def gmbl_dataset_upload_pipeline(
    dataset_path: str,
    repo_id: str,
) -> int:
    """
    Upload GMBL text dataset to HuggingFace Hub

    Args:
        dataset_path: Path to merged dataset.json file
        repo_id: HuggingFace repo ID (e.g., "your-username/geometry-gmbl")

    Returns:
        Number of samples uploaded
    """
    from llm_engineering import settings

    num_uploaded = upload_gmbl_to_huggingface(
        dataset_path=dataset_path,
        repo_id=repo_id,
        token=settings.HF_TOKEN,
    )

    logger.info(f"GMBL dataset upload completed: {num_uploaded} samples")
    return num_uploaded
