# from zenml import step
# from datasets import Dataset
# import json
# from loguru import logger


# @step
# def load_dataset_step(
#     train_path: str,
#     eval_path: str = None
# ) -> tuple[Dataset, Dataset]:
#     """Load training and evaluation datasets"""

#     logger.info(f"Loading dataset from {train_path}")

#     with open(train_path) as f:
#         train_data = json.load(f)

#     train_dataset = Dataset.from_list(train_data)
#     logger.info(f"Train samples: {len(train_dataset)}")

#     eval_dataset = None
#     if eval_path:
#         with open(eval_path) as f:
#             eval_data = json.load(f)
#         eval_dataset = Dataset.from_list(eval_data)
#         logger.info(f"Eval samples: {len(eval_dataset)}")

#     return train_dataset, eval_dataset
