from zenml import pipeline
from loguru import logger

from steps.dataset import (
    load_source_data,
    create_prompts,
    generate_gmbl_dataset,
    save_dataset_to_json
)

@pipeline
def dataset_generation_pipeline(
    source_json_path: str = "./data/data-1k/diagram_train_1k_vn.json",
    test_size: float = 0.2,
    save_json: bool = True,
    output_dir: str = "./data/generated_gmbl"
):
    logger.info("Starting GMBL dataset generation pipeline")

    documents = load_source_data(source_json_path=source_json_path)

    prompts = create_prompts(documents=documents)

    train_test_split = generate_gmbl_dataset(
        prompts=prompts,
        test_size=test_size
    )

    if save_json:
        dataset_dir = save_dataset_to_json(
            train_test_split=train_test_split,
            output_dir=output_dir
        )
        logger.success(f"Dataset saved to: {dataset_dir}")
        return dataset_dir

    logger.success(f"Dataset generation completed")
    return train_test_split
