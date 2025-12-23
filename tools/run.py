import click
from pathlib import Path
from loguru import logger
from datetime import datetime as dt
import yaml
import json

from llm_engineering.domains.training_config import TrainingConfig
from llm_engineering.settings import settings
# from pipelines.figure_extraction import figure_extraction_pipeline
from pipelines.dataset_upload import dataset_upload_pipeline
from pipelines.training import training_pipeline
from pipelines.inference import inference_pipeline


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
@click.option(
    "--encode-images",
    is_flag=True,
    default=False,
    help="Encode images to tokens (run once)"
)
@click.option(
    "--run-inference",
    is_flag=True,
    default=False,
    help="Run inference pipeline to test trained model"
)
def main(
    no_cache: bool = False,
    run_upload_dataset: bool = False,
    run_train: bool = False,
    encode_images: bool = False,
    run_inference: bool = False,
) -> None:
    assert run_upload_dataset or run_train or encode_images or run_inference, "Please use one of the options"

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

    if encode_images:
        config_path = root_dir / "configs" / "training.yaml"
        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        input_json = str(root_dir / config_data["dataset_path"])
        images_dir = str(root_dir / config_data["images_dir"])
        output_json = input_json.replace(".json", "_cached.json")

        logger.info(f"Encoding: {input_json} -> {output_json}")

        from llm_engineering.applications.training.images_encoder import encode_dataset_images
        encode_dataset_images(input_json, output_json, images_dir, config_data["vq_model_path"])

        logger.success(f"Done! Update dataset_path: {Path(output_json).relative_to(root_dir)}")

    if run_train:
        config_path = root_dir / "configs" / "training.yaml"
        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        config = TrainingConfig(**config_data)

        dataset_path = str(root_dir / config_data["dataset_path"])
        images_dir = str(root_dir / config_data.get("images_dir", ""))

        merge_model = config_data.get("merge_model", True)
        push_to_hub = config_data.get("push_to_hub", False)
        hf_repo_name = config_data.get("hf_repo_name")
        test_image = config_data.get("test_image")

        if test_image:
            test_image = str(root_dir / test_image)
            if not Path(test_image).exists():
                test_image = None

        hf_token = settings.HF_TOKEN if push_to_hub else None
        if push_to_hub and not hf_token:
            push_to_hub = False

        pipeline_args["run_name"] = f"training_run_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"

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

    if run_inference:
        config_path = root_dir / "configs" / "inference.yaml"
        assert config_path.exists(), f"Config file not found: {config_path}"

        logger.info(f"Loading inference config from {config_path}")
        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        # Empty model_path means use base Qwen model
        model_path = config_data.get("model_path", "")
        if model_path:
            model_path = str(root_dir / model_path)
            if not Path(model_path).exists():
                logger.warning(f"Trained model not found at {model_path}, will use base Qwen model")
                model_path = ""

        test_prompt = config_data.get("test_prompt", "Vẽ đoạn thẳng AB có độ dài 5")
        output_dir = str(root_dir / config_data.get("output_dir", "./outputs/inference_test"))

        pipeline_args["run_name"] = f"inference_run_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"

        logger.info("Starting inference pipeline")
        logger.info(f"Model: {model_path or 'Base Qwen2.5-3B-Instruct'}")
        logger.info(f"Prompt: {test_prompt}")

        inference_pipeline.with_options(**pipeline_args)(
            model_path=model_path,
            vq_model_path=config_data["vq_model_path"],
            test_prompt=test_prompt,
            output_dir=output_dir
        )

if __name__ == "__main__":
    main()