from typing import Annotated
from pathlib import Path
import json

from zenml import step
from loguru import logger

from pipeline.services.datasets.generation import InstructiveDatasetGenerator
from pipeline.domain import Document
from pipeline.domain import GenerateDatasetSamplesPrompt
from pipeline.domain import InstructTrainTestSplit

@step
def load_source_data(
    source_json_path: str,
) -> Annotated[list[Document], "documents"]:
    source_path = Path(source_json_path)

    if not source_path.exists():
        raise FileNotFoundError(f"Source data not found: {source_json_path}")

    with open(source_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    for item in data:
        if "caption_vn" in item:
            doc = Document(
                caption=item.get("caption", [""])[0],
                image_dir=item.get("image", ""),
                caption_vn=item["caption_vn"]
            )
            documents.append(doc)

    logger.info(f"Loaded {len(documents)} documents from {source_json_path}")
    return documents

@step
def create_prompts(
    documents: list[Document],
) -> Annotated[list[GenerateDatasetSamplesPrompt], "prompts"]:
    """Create prompts for LLM generation"""
    prompts = []

    for doc in documents:
        prompt = InstructiveDatasetGenerator.get_prompt(doc)
        prompts.append(prompt)

    logger.info(f"Created {len(prompts)} prompts")
    return prompts


@step
def generate_gmbl_dataset(
    prompts: list[GenerateDatasetSamplesPrompt],
    test_size: float = 0.2,
    batch_size: int = 4,
) -> Annotated[InstructTrainTestSplit, "train_test_split"]:


    logger.info(f"Generating dataset from {len(prompts)} prompts with batch_size={batch_size}...")

    train_test_split = InstructiveDatasetGenerator.generate(
        prompts=prompts,
        test_size=test_size,
        batch_size=batch_size
    )

    logger.info(f"Generated {train_test_split.train.num_samples} train samples")
    logger.info(f"Generated {train_test_split.test.num_samples} test samples")

    return train_test_split


@step
def save_dataset_to_json(
    train_test_split: InstructTrainTestSplit,
    output_dir: str
) -> Annotated[str, "output_path"]:

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_path = output_path / "train.json"
    test_path = output_path / "test.json"

    # Load existing data if file exists
    existing_train = []
    existing_test = []

    if train_path.exists():
        with open(train_path, "r", encoding="utf-8") as f:
            existing_train = json.load(f)

    if test_path.exists():
        with open(test_path, "r", encoding="utf-8") as f:
            existing_test = json.load(f)

    train_data = [sample.model_dump() for sample in train_test_split.train.samples]
    test_data = [sample.model_dump() for sample in train_test_split.test.samples]

    # Merge old + new data
    merged_train = existing_train + train_data
    merged_test = existing_test + test_data

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(merged_train, f, ensure_ascii=False, indent=2)

    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(merged_test, f, ensure_ascii=False, indent=2)

    logger.success(f"Saved train dataset to {train_path}")
    logger.success(f"Saved test dataset to {test_path}")

    return str(output_path)

