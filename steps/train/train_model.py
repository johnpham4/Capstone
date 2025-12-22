from pathlib import Path
from zenml import step
from loguru import logger
from typing import Optional

from llm_engineering.applications.training.model_trainer import ModelTrainer


@step
def train_step(
    trainer: ModelTrainer,
    train_data: str,
    images_dir: str,
    eval_data: Optional[str] = None
) -> str:
    """Load datasets and execute training"""

    logger.info("Loading datasets")
    train_dataset, eval_dataset = trainer.load_dataset(train_data, images_dir, eval_data)

    logger.info("Starting training")
    hf_trainer = trainer.train(train_dataset, eval_dataset)

    output_path = Path(trainer.config.output_dir) / "final"
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving model to {output_path}")
    hf_trainer.save_model(str(output_path))
    trainer.tokenizer.save_pretrained(str(output_path))

    logger.success(f"Model saved to {output_path}")

    return str(output_path)
