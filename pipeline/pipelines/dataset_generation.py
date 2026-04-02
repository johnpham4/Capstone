from zenml import pipeline
from loguru import logger
import os
import yaml


from pipeline.steps.dataset import (
    load_source_data,
    create_prompts,
    generate_gmbl_dataset,
    save_dataset_to_json
)

@pipeline
def dataset_generation_pipeline(
    config_path: str = "pipeline/configs/dataset_generation.yaml",
    source_json_path: str = "",
    output_dir: str = "",
    test_size: float = 0.2,
    batch_size: int = 4,
    sleep_seconds: float = 2.0,
    log_every_batches: int = 10,
    max_concurrency: int = 4,
    enable_dsl_validation: bool = True,
    save_json: bool = True,
):
    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        params = config.get("parameters", {})

        source_json_path = params.get("source_json_path", source_json_path)
        output_dir = params.get("output_dir", output_dir)

    logger.info("Starting GMBL dataset generation pipeline")

    documents = load_source_data(source_json_path=source_json_path)

    prompts = create_prompts(documents=documents)

    train_test_split = generate_gmbl_dataset(
        prompts=prompts,
        test_size=test_size,
        batch_size=batch_size,
        sleep_seconds=sleep_seconds,
        log_every_batches=log_every_batches,
        max_concurrency=max_concurrency,
        enable_dsl_validation=enable_dsl_validation,
    )

    if save_json:
        dataset_dir = save_dataset_to_json(
            train_test_split=train_test_split,
            output_dir=output_dir
        )
        logger.success(f"Dataset saved to: {dataset_dir}")
        return dataset_dir

    logger.success("Dataset generation completed")
    return train_test_split

