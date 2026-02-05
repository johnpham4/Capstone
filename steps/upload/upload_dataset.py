from typing import Annotated, Union
from pathlib import Path
import json

from zenml import step
from loguru import logger
from datasets import Dataset, DatasetDict, Features, Value, Image as HFImage

from src.models.domain.dataset import TrainTestSplit

@step
def upload_to_huggingface(
    dataset: Union[TrainTestSplit, str],
    repo_id: str,
    token: str
) -> Annotated[int, "num_uploaded"]:

    if isinstance(dataset, str):
        dataset_path = Path(dataset)

        train_path = dataset_path / "train.json"
        test_path = dataset_path / "test.json"

        if not train_path.exists():
            raise FileNotFoundError(f"Train dataset not found: {train_path}")

        with open(train_path, "r", encoding="utf-8") as f:
            train_data = json.load(f)

        test_data = []
        if test_path.exists():
            with open(test_path, "r", encoding="utf-8") as f:
                test_data = json.load(f)

        features = Features({
            "image": HFImage(),
            "instruction": Value("string"),
            "output": Value("string")
        })

        train_dataset = Dataset.from_dict({
            "image": [str(dataset_path / d["image_dir"]) for d in train_data],
            "instruction": [d["instruction"] for d in train_data],
            "output": [d["answer"] for d in train_data]
        }, features=features)

        total_samples = len(train_dataset)

        if test_data:
            test_dataset = Dataset.from_dict({
                "image": [str(dataset_path / d["image_dir"]) for d in test_data],
                "instruction": [d["instruction"] for d in test_data],
                "output": [d["answer"] for d in test_data]
            }, features=features)

            total_samples += len(test_dataset)

            dataset_dict = DatasetDict({
                "train": train_dataset,
                "test": test_dataset
            })

            logger.info(f"Uploading {total_samples} samples (train: {len(train_dataset)}, test: {len(test_dataset)}) to {repo_id}")
        else:
            dataset_dict = DatasetDict({"train": train_dataset})
            logger.info(f"Uploading {total_samples} samples (train only) to {repo_id}")

    else:
        dataset_dict = dataset.to_huggingface(include_images=True)
        total_samples = len(dataset_dict["train"]) + len(dataset_dict["test"])
        logger.info(f"Uploading {total_samples} samples (train: {len(dataset_dict['train'])}, test: {len(dataset_dict['test'])}) to {repo_id}")

    try:
        # Create README with dataset card
        readme_content = f"""---
dataset_info:
  features:
  - name: image
    dtype: image
  - name: instruction
    dtype: string
  - name: output
    dtype: string
  splits:
  - name: train
    num_examples: {len(dataset_dict['train'])}
"""
        if 'test' in dataset_dict:
            readme_content += f"""  - name: test
    num_examples: {len(dataset_dict['test'])}
"""

        readme_content += """
tags:
- geometry
- vision-language
- multimodal
task_categories:
- visual-question-answering
language:
- en
size_categories:
- 1K<n<10K
---

# Geometry Dataset

This dataset contains geometry problems with diagrams for vision-language model training.

## Dataset Structure

- **image**: Geometry diagram image
- **instruction**: Problem statement or question
- **output**: Answer or solution
"""

        dataset_dict.push_to_hub(
            repo_id=repo_id,
            token=token,
        )

        # Upload README separately
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_file(
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )

        logger.success(f"Successfully uploaded dataset with README")
        return total_samples

    except Exception as e:
        logger.error(f"Failed to upload dataset: {e}")
        raise
