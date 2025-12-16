# """ZenML training pipeline for GeoUni finetuning"""

# from zenml import pipeline
# from loguru import logger

# from llm_engineering.domains.training_config import TrainingConfig
# from steps.train.load_model import load_model_step
# from steps.train.load_dataset import load_dataset_step
# from steps.train.train_model import train_step


# @pipeline
# def training_pipeline(
#     config: TrainingConfig,
#     train_data: str,
#     eval_data: str = None
# ):
#     """ZenML training pipeline"""

#     logger.info("Starting ZenML training pipeline")

#     # Steps
#     trainer = load_model_step(config)
#     train_dataset, eval_dataset = load_dataset_step(train_data, eval_data)
#     model_path = train_step(trainer, train_dataset, eval_dataset)

#     logger.success("Pipeline completed!")

#     return model_path
