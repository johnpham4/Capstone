from typing import Annotated
from pathlib import Path
import json

from zenml import step
from datasets import Dataset, DatasetDict
from loguru import logger


@step
def upload_gmbl_to_huggingface(
    dataset_path: str,
    repo_id: str,
    token: str,
) -> Annotated[int, "num_uploaded"]:
    """Upload GMBL text dataset to HuggingFace Hub"""

    dataset_file = Path(dataset_path)

    if not dataset_file.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        return 0

    # Load merged dataset
    with open(dataset_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Create HuggingFace DatasetDict
    dataset_dict = {}

    for split_name in ["train", "test"]:
        if split_name in data:
            split_data = data[split_name]

            # Extract instructions and answers
            instructions = [item["instruction"] for item in split_data]
            answers = [item["answer"] for item in split_data]

            dataset_dict[split_name] = Dataset.from_dict({
                "instruction": instructions,
                "output": answers
            })

            logger.info(f"Prepared {len(split_data)} samples for {split_name} split")

    if not dataset_dict:
        logger.warning("No data to upload")
        return 0

    hf_dataset = DatasetDict(dataset_dict)

    try:
        logger.info(f"Uploading to {repo_id}...")
        hf_dataset.push_to_hub(
            repo_id=repo_id,
            token=token,
            private=False,
        )

        total_samples = sum(len(ds) for ds in dataset_dict.values())
        logger.success(f"Successfully uploaded {total_samples} total samples")
        return total_samples

    except Exception as e:
        logger.error(f"Failed to upload dataset: {e}")
        raise
