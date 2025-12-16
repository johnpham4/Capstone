from typing import Annotated
from pathlib import Path
import json

from zenml import step
from datasets import Dataset, Image as HFImage, Features, Value
from huggingface_hub import HfApi
from loguru import logger


@step
def upload_to_huggingface(
    dataset_path: str,
    repo_id: str,
    token: str,
    split: str = "train"
) -> Annotated[int, "num_uploaded"]:
    """Upload figure dataset to HuggingFace Hub"""

    dataset_dir = Path(dataset_path)
    if not dataset_dir.exists():
        logger.error(f"Dataset path not found: {dataset_path}")
        return 0

    data = []
    skipped = 0

    # Collect figure metadata
    for json_file in dataset_dir.rglob("*_figures.json"):
        logger.info(f"Found JSON: {json_file}")

        with open(json_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        source = metadata.get("name", json_file.stem)

        for fig in metadata.get("figures", []):
            img_path = Path(fig.get("image_path", ""))

            if img_path.exists():
                data.append({
                    "image": str(img_path),   # path → HF Image
                    "caption": fig.get("caption", ""),
                    "source": source,
                })
            else:
                skipped += 1
                logger.debug(f"Image not found: {img_path}")

    if not data:
        logger.warning(f"No figures found in {dataset_path}")
        return 0

    logger.info(f"Collected {len(data)} figures (skipped {skipped})")

    # Define dataset schema
    features = Features({
        "image": HFImage(),
        "caption": Value("string"),
        "source": Value("string"),
    })

    # Create HF Dataset
    dataset = Dataset.from_list(data, features=features)

    try:
        logger.info(f"Uploading {len(dataset)} samples to {repo_id}")
        dataset.push_to_hub(
            repo_id=repo_id,
            token=token,
            split=split,
        )

        logger.success(f"Successfully uploaded {len(dataset)} figures")
        return len(dataset)

    except Exception as e:
        logger.error(f"Failed to upload dataset: {e}")
        raise
