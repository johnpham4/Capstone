from typing import Annotated, Union
from pathlib import Path
import json

from zenml import step
from loguru import logger
from datasets import Dataset, DatasetDict, Features, Value, Image as HFImage

from llm_engineering.domains.dataset import TrainTestSplit

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
            "image": [d["image_dir"] for d in train_data],
            "instruction": [d["instruction"] for d in train_data],
            "output": [d["answer"] for d in train_data]
        }, features=features)

        total_samples = len(train_dataset)

        if test_data:
            test_dataset = Dataset.from_dict({
                "image": [d["image_dir"] for d in test_data],
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
        dataset_dict.push_to_hub(
            repo_id=repo_id,
            token=token,
        )

        logger.success(f"Successfully uploaded dataset")
        return total_samples

    except Exception as e:
        logger.error(f"Failed to upload dataset: {e}")
        raise
