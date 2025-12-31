from zenml import pipeline
from loguru import logger

from steps.dataset.generate_dataset import (
    load_source_data,
    create_prompts,
    generate_gmbl_dataset,
    merge_datasets
)


@pipeline
def dataset_generation_pipeline(
    source_json_path: str = "./data/data-1k/diagram_train_1k_vn.json",
    test_size: float = 0.2,
) -> str:

    logger.info("Starting GMBL dataset generation pipeline")

    documents = load_source_data(source_json_path=source_json_path)

    prompts = create_prompts(documents=documents)

    dataset_dir = generate_gmbl_dataset(
        prompts=prompts,
        test_size=test_size
    )

    merged_path = merge_datasets(dataset_dir=dataset_dir)

    logger.success(f"Dataset generation completed: {merged_path}")
    return merged_path
