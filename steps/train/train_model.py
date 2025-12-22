from pathlib import Path
from zenml import step
from datasets import Dataset
from loguru import logger

from llm_engineering.applications.training.model_trainer import ModelTrainer


@step
def train_step(
    trainer: ModelTrainer,
    train_dataset: Dataset,
    eval_dataset: Dataset = None
) -> str:
    """Execute training"""

    logger.info("Starting training")

    hf_trainer = trainer.train(train_dataset, eval_dataset)

    output_path = Path(trainer.config.output_dir) / "final"
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving model to {output_path}")
    hf_trainer.save_model(str(output_path))
    trainer.tokenizer.save_pretrained(str(output_path))

    logger.success(f"Model saved to {output_path}")

    return str(output_path)
