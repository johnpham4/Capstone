from zenml import pipeline
from loguru import logger

from pipeline.steps.dataset import (
    load_source_data_for_question_generation,
    create_question_prompts,
    generate_question_dataset,
    save_question_dataset_to_json,
)


@pipeline
def question_generation_pipeline(
    source_json_path: str = "./dataset/data/diagrams_filter.json",
    test_size: float = 0.2,
    batch_size: int = 4,
    sleep_seconds: float = 2.0,
    log_every_batches: int = 10,
    max_concurrency: int = 4,
    enable_dsl_validation: bool = False,
    save_json: bool = True,
    output_dir: str = "./dataset/data",
):
    logger.info("Starting question generation pipeline")

    documents = load_source_data_for_question_generation(source_json_path=source_json_path)

    prompts = create_question_prompts(documents=documents)

    train_test_split = generate_question_dataset(
        prompts=prompts,
        test_size=test_size,
        batch_size=batch_size,
        sleep_seconds=sleep_seconds,
        log_every_batches=log_every_batches,
        max_concurrency=max_concurrency,
        enable_dsl_validation=enable_dsl_validation,
    )

    if save_json:
        dataset_dir = save_question_dataset_to_json(
            train_test_split=train_test_split,
            output_dir=output_dir,
        )
        logger.success(f"Question dataset saved to: {dataset_dir}")
        return dataset_dir

    logger.success("Question generation completed")
    return train_test_split
