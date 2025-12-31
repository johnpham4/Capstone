from enum import Enum

from llm_engineering.domains.types import DataCategory
from loguru import logger

try:
    from datasets import Dataset, DatasetDict, concatenate_datasets
except ImportError:
    logger.warning("Huggingface datasets not installed. Install with `pip install datasets`")

from llm_engineering.domains.orm.nosql import NoSQLBaseDocument

class InstructDatasetSample(NoSQLBaseDocument):
    instruction: str
    answer: str

    class Settings:
        name: str = DataCategory.INSTRUCT_DATASET_SAMPLES

class InstructDataset(NoSQLBaseDocument):
    samples: list[InstructDatasetSample]

    class Config:
        category = DataCategory.INSTRUCT_DATASET

    @property
    def num_samples(self) -> int:
        return len(self.samples)

    def to_huggingface(self) -> "Dataset":
        data = [sample.model_dump() for sample in self.samples]

        return Dataset.from_dict(
            {"instruction": [d["instruction"] for d in data], "output": [d["answer"] for d in data]}
        )


class TrainTestSplit(NoSQLBaseDocument):
    train: InstructDataset
    test: InstructDataset
    test_split_size: float

    def to_huggingface(self, flatten: bool = False) -> "DatasetDict":
        train_datasets = self.train.to_huggingface()
        test_datasets = self.test.to_huggingface()

        if flatten:
            train_datasets = concatenate_datasets(list(train_datasets.values()))
            test_datasets = concatenate_datasets(list(test_datasets.values()))
        else:
            train_datasets = Dataset.from_dict(train_datasets)
            test_datasets = Dataset.from_dict(test_datasets)

        return DatasetDict({"train": train_datasets, "test": test_datasets})


class InstructTrainTestSplit(TrainTestSplit):
    train: InstructDataset
    test: InstructDataset
    test_split_size: float

    class Config:
        category = DataCategory.INSTRUCT_DATASET
