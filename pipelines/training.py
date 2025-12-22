from zenml import pipeline
from loguru import logger

from llm_engineering.domains.training_config import TrainingConfig
from steps.train.load_model import load_model_step
from steps.train.load_dataset import load_dataset_step
from steps.train.train_model import train_step
from steps.train.merge_model import merge_lora_step
from steps.train.push_model import push_to_hub_step
from steps.train.test_inference import test_inference_step


@pipeline
def training_pipeline(
    config: TrainingConfig,
    train_data: str,
    images_dir: str,
    eval_data: str = None,
    merge_model: bool = True,
    push_to_hub: bool = False,
    hf_repo_name: str = None,
    hf_token: str = None,
    test_image: str = None
):
    """Complete training pipeline with merge, push, and test"""

    logger.info("Starting training pipeline")

    trainer = load_model_step(config)
    train_dataset, eval_dataset = load_dataset_step(train_data, images_dir, eval_data)
    checkpoint_path = train_step(trainer, train_dataset, eval_dataset)

    if merge_model:
        logger.info("Merging LoRA adapter")
        merged_path = merge_lora_step(
            checkpoint_path=checkpoint_path,
            base_model_path=config.base_llm_path,
            output_path=f"{config.output_dir}/merged"
        )

        if push_to_hub and hf_repo_name and hf_token:
            logger.info("Pushing to HuggingFace Hub")
            hub_url = push_to_hub_step(
                model_path=merged_path,
                repo_name=hf_repo_name,
                hf_token=hf_token,
                private=False
            )

        if test_image:
            logger.info("Running inference test")
            test_result = test_inference_step(
                model_path=merged_path,
                vq_model_path=config.vq_model_path,
                test_image_path=test_image,
                output_dir=f"{config.output_dir}/test_output"
            )

    logger.success("Pipeline completed")

    return checkpoint_path
