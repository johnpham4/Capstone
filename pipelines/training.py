from zenml import pipeline
from loguru import logger

from llm_engineering.domains.training_config import TrainingConfig
from steps.train.load_model import load_model_step
from steps.train.load_dataset import load_dataset_step
from steps.train.train_model import train_step


@pipeline
def training_pipeline(
    config: TrainingConfig,
    train_data: str,
    images_dir: str,
    eval_data: str = None
):
    """Training pipeline for GeoUni line segment generation"""

    logger.info("Starting training pipeline")

    trainer = load_model_step(config)
    train_dataset, eval_dataset = load_dataset_step(train_data, images_dir, eval_data)
    model_path = train_step(trainer, train_dataset, eval_dataset)

    logger.success("Pipeline completed")

    return model_path
