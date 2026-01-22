from enum import Enum

from llm_src.domains.types import DataCategory
from loguru import logger
from pydantic import BaseModel

try:
    from datasets import Dataset, DatasetDict, concatenate_datasets
except ImportError:
    logger.warning("Huggingface datasets not installed. Install with `pip install datasets`")

from llm_src.domains.odm.nosql import NoSQLBaseDocument

class InstructDatasetSample(BaseModel):
    image_dir: str
    instruction: str
    answer: str

    class Settings:
        name: str = DataCategory.INSTRUCT_DATASET_SAMPLES

class InstructDataset(BaseModel):
    samples: list[InstructDatasetSample]

    class Config:
        category = DataCategory.INSTRUCT_DATASET

    @property
    def num_samples(self) -> int:
        return len(self.samples)

    def to_huggingface(self, include_images: bool = False) -> "Dataset":
        from datasets import Image as HFImage, Features, Value

        data = [sample.model_dump() for sample in self.samples]

        dataset_dict = {
            "instruction": [d["instruction"] for d in data],
            "output": [d["answer"] for d in data]
        }

        if include_images:
            dataset_dict["image"] = [d["image_dir"] for d in data]
            features = Features({
                "image": HFImage(),
                "instruction": Value("string"),
                "output": Value("string")
            })
            return Dataset.from_dict(dataset_dict, features=features)

        return Dataset.from_dict(dataset_dict)


class TrainTestSplit(BaseModel):
    train: InstructDataset
    test: InstructDataset
    test_split_size: float

    def to_huggingface(self, flatten: bool = False, include_images: bool = False) -> "DatasetDict":
        train_datasets = self.train.to_huggingface(include_images=include_images)
        test_datasets = self.test.to_huggingface(include_images=include_images)

        if flatten:
            train_datasets = concatenate_datasets(list(train_datasets.values()))
            test_datasets = concatenate_datasets(list(test_datasets.values()))

        return DatasetDict({"train": train_datasets, "test": test_datasets})


class InstructTrainTestSplit(BaseModel):
    train: InstructDataset
    test: InstructDataset
    test_split_size: float

    class Config:
        category = DataCategory.INSTRUCT_DATASET
