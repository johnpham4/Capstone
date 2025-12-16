# from zenml import step
# from loguru import logger

# from llm_engineering.domains.training_config import TrainingConfig
# from llm_engineering.applications.training.model_trainer import ModelTrainer


# @step
# def load_model_step(config: TrainingConfig) -> ModelTrainer:
#     """Load base model and apply LoRA"""

#     logger.info("Loading model...")
#     trainer = ModelTrainer(config)
#     trainer.load_model()

#     return trainer
