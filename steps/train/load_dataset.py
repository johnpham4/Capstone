from zenml import step
from datasets import Dataset
import json
from loguru import logger
from typing import Optional


@step
def load_dataset_step(
    train_path: str,
    images_dir: str,
    eval_path: Optional[str] = None
) -> tuple[Dataset, Optional[Dataset]]:
    """Load training and evaluation datasets"""

    logger.info(f"Loading dataset from {train_path}")

    with open(train_path, encoding='utf-8') as f:
        train_data = json.load(f)

    for sample in train_data:
        sample['images_dir'] = images_dir

    train_dataset = Dataset.from_list(train_data)
    logger.info(f"Train samples: {len(train_dataset)}")

    eval_dataset = None
    if eval_path:
        with open(eval_path, encoding='utf-8') as f:
            eval_data = json.load(f)
        for sample in eval_data:
            sample['images_dir'] = images_dir
        eval_dataset = Dataset.from_list(eval_data)
        logger.info(f"Eval samples: {len(eval_dataset)}")

    return train_dataset, eval_dataset
