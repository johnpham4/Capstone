import click
from pathlib import Path
from loguru import logger
from datetime import datetime as dt
import yaml

from llm_engineering.domains.training_config import TrainingConfig
from llm_engineering.settings import settings
from pipelines.figure_extraction import figure_extraction_pipeline
from pipelines.dataset_upload import dataset_upload_pipeline
from pipelines.training import training_pipeline


@click.command()
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Disable caching for the pipeline run.",
)
# @click.option(
#     "--run-extract",
#     is_flag=True,
#     default=False,
#     help="Run extraction pipeline (figures + text)"
# )
@click.option(
    "--run-upload-dataset",
    is_flag=True,
    default=False,
    help="Upload dataset to HuggingFace Hub"
)
@click.option(
    "--run-train",
    is_flag=True,
    default=False,
    help="Run training pipeline"
)
def main(
    no_cache: bool = False,
    # run_extract: bool = False,
    run_upload_dataset: bool = False,
    run_train: bool = False,
) -> None:
    assert run_upload_dataset or run_train, "Please use one of the options"

    pipeline_args = {"enable_cache": not no_cache}
    root_dir = Path(__file__).resolve().parent.parent

    # if run_extract:
    #     pipeline_args["config_path"] = root_dir / "configs" / "figure_extraction.yaml"
    #     assert pipeline_args["config_path"].exists(), f"Config file not found: {pipeline_args['config_path']}"
    #     pipeline_args["run_name"] = f"extract_pipeline_run_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"

    #     logger.info("Starting extraction pipeline (figures + text)")
    #     figure_extraction_pipeline.with_options(**pipeline_args)()

    if run_upload_dataset:
        pipeline_args["config_path"] = root_dir / "configs" / "dataset_upload.yaml"
        assert pipeline_args["config_path"].exists(), f"Config file not found: {pipeline_args['config_path']}"
        pipeline_args["run_name"] = f"upload_dataset_run_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"

        assert settings.HF_TOKEN, "HuggingFace token required. Set HF_TOKEN in .env"
        logger.info("Starting dataset upload pipeline")
        dataset_upload_pipeline.with_options(**pipeline_args)()

    if run_train:
        config_path = root_dir / "configs" / "training.yaml"
        assert config_path.exists(), f"Config file not found: {config_path}"

        logger.info(f"Loading config from {config_path}")
        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        config = TrainingConfig(
            vq_model_path=config_data["vq_model_path"],
            base_llm_path=config_data["base_llm_path"],
            output_dir=config_data["output_dir"],
            batch_size=config_data.get("batch_size", 1),
            epochs=config_data.get("epochs", 3),
            learning_rate=config_data.get("learning_rate", 2e-4),
            lora_r=config_data.get("lora_r", 8),
            lora_alpha=config_data.get("lora_alpha", 16),
            gradient_accumulation_steps=config_data.get("gradient_accumulation_steps", 4),
            use_8bit=config_data.get("use_8bit", True),
            gradient_checkpointing=config_data.get("gradient_checkpointing", True),
            fp16=config_data.get("fp16", True),
        )

        dataset_path = str(root_dir / config_data["dataset_path"])
        images_dir = str(root_dir / config_data["images_dir"])

        assert Path(dataset_path).exists(), f"Dataset not found: {dataset_path}"
        assert Path(images_dir).exists(), f"Images directory not found: {images_dir}"

        merge_model = config_data.get("merge_model", True)
        push_to_hub = config_data.get("push_to_hub", False)
        hf_repo_name = config_data.get("hf_repo_name")
        test_image = config_data.get("test_image")

        if test_image:
            test_image = str(root_dir / test_image)
            if not Path(test_image).exists():
                logger.warning(f"Test image not found: {test_image}")
                test_image = None

        hf_token = settings.HF_TOKEN if push_to_hub else None
        if push_to_hub and not hf_token:
            logger.warning("HF_TOKEN not set, skipping push to hub")
            push_to_hub = False

        pipeline_args["run_name"] = f"training_run_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"

        logger.info("Starting training pipeline")
        logger.info(f"Merge model: {merge_model}")
        logger.info(f"Push to hub: {push_to_hub}")
        logger.info(f"Test inference: {test_image is not None}")

        training_pipeline.with_options(**pipeline_args)(
            config=config,
            train_data=dataset_path,
            images_dir=images_dir,
            eval_data=None,
            merge_model=merge_model,
            push_to_hub=push_to_hub,
            hf_repo_name=hf_repo_name,
            hf_token=hf_token,
            test_image=test_image
        )


if __name__ == "__main__":
    main()