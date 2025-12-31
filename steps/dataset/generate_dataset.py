from typing import Annotated
from pathlib import Path
import json

from zenml import step
from loguru import logger

from llm_engineering.applications.datasets.generation import InstructiveDatasetGenerator
from llm_engineering.domains.documents import Document
from llm_engineering.domains.prompt import GenerateDatasetSamplesPrompt


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
) -> Annotated[str, "output_path"]:
    """Generate GMBL dataset using LLM"""
    logger.info(f"Generating dataset from {len(prompts)} prompts...")

    train_test_split = InstructiveDatasetGenerator.generate(
        prompts=prompts,
        test_size=test_size
    )

    logger.info(f"Generated {train_test_split.train.num_samples} train samples")
    logger.info(f"Generated {train_test_split.test.num_samples} test samples")

    # Save to JSON
    output_dir = Path("./data/generated_gmbl")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.json"
    test_path = output_dir / "test.json"

    train_data = [sample.model_dump() for sample in train_test_split.train.samples]
    test_data = [sample.model_dump() for sample in train_test_split.test.samples]

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    logger.success(f"Saved train dataset to {train_path}")
    logger.success(f"Saved test dataset to {test_path}")

    return str(output_dir)


@step
def merge_datasets(
    dataset_dir: str,
) -> Annotated[str, "merged_path"]:
    """Merge train and test datasets for upload"""
    dataset_path = Path(dataset_dir)

    train_path = dataset_path / "train.json"
    test_path = dataset_path / "test.json"

    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # Create HuggingFace format
    hf_dataset = {
        "train": train_data,
        "test": test_data
    }

    merged_path = dataset_path / "dataset.json"
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(hf_dataset, f, ensure_ascii=False, indent=2)

    logger.success(f"Merged dataset saved to {merged_path}")
    return str(merged_path)
