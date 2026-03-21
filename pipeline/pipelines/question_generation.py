from zenml import pipeline
from loguru import logger

from pipeline.steps.dataset import (
    load_source_data_for_question_generation,
    generate_question_dataset,
    save_question_dataset_to_json,
)


@pipeline
def question_generation_pipeline(
    source_json_path: str = "./dataset/data/diagrams_filter.json",
    batch_size: int = 4,
    sleep_seconds: float = 2.0,
    log_every_batches: int = 10,
    max_concurrency: int = 4,
    save_json: bool = True,
):
    logger.info("Starting question generation pipeline")

    source_items = load_source_data_for_question_generation(source_json_path=source_json_path)

    updated_items = generate_question_dataset(
        source_items=source_items,
        batch_size=batch_size,
        sleep_seconds=sleep_seconds,
        log_every_batches=log_every_batches,
        max_concurrency=max_concurrency,
    )

    if save_json:
        dataset_dir = save_question_dataset_to_json(
            updated_items=updated_items,
            source_json_path=source_json_path,
        )
        logger.success(f"Question dataset saved to: {dataset_dir}")
        return dataset_dir

    logger.success("Question generation completed")
    return updated_items
