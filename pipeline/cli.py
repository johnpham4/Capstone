import click
import subprocess
import sys
from importlib import import_module
from pathlib import Path
from loguru import logger
from datetime import datetime as dt
from src.config.settings.base import settings

from pipeline.pipelines.data_preparation import data_preparation_pipeline
from pipeline.pipelines.dataset_generation import dataset_generation_pipeline
from pipeline.pipelines.dataset_upload import dataset_upload_pipeline

@click.command()
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Disable caching for the pipeline run.",
)
@click.option(
    "--run-prepare-data",
    is_flag=True,
    default=False,
    help="Prepare SynthGeo dataset with Vietnamese translations"
)
@click.option(
    "--run-upload-dataset",
    is_flag=True,
    default=False,
    help="Upload dataset to HuggingFace Hub"
)
@click.option(
    "--run-generate-gmbl",
    is_flag=True,
    default=False,
    help="Generate GMBL dataset from Vietnamese captions"
)
@click.option(
    "--run-finetune",
    is_flag=True,
    default=False,
    help="Run finetuning on AWS SageMaker"
)
@click.option(
    "--run-render-diagram",
    is_flag=True,
    default=False,
    help="Render diagram images from generated DSL dataset"
)
@click.option(
    "--num-epochs",
    type=int,
    default=1,
    help="Number of training epochs"
)
@click.option(
    "--batch-size",
    type=int,
    default=2,
    help="Batch size per GPU device"
)
@click.option(
    "--learning-rate",
    type=float,
    default=2e-4,
    help="Learning rate"
)
@click.option(
    "--dataset-workspace",
    type=str,
    default="minn4",
    help="Hugging Face workspace containing the dataset"
)
def main(
    no_cache: bool = False,
    run_prepare_data: bool = False,
    run_upload_dataset: bool = False,
    run_generate_gmbl: bool = False,
    run_finetune: bool = False,
    run_render_diagram: bool = False,
    num_epochs: int = 1,
    batch_size: int = 2,
    learning_rate: float = 2e-4,
    dataset_workspace: str = "minn4",
) -> None:
    assert (
        run_prepare_data
        or run_upload_dataset
        or run_generate_gmbl
        or run_finetune
        or run_render_diagram
    ), "Please use one of the options"

    pipeline_args = {"enable_cache": not no_cache}
    root_dir = Path(__file__).resolve().parent.parent

    if run_prepare_data:
        pipeline_args["config_path"] = str(root_dir / "configs" / "data_preparation.yaml")
        assert pipeline_args["config_path"].exists(), f"Config file not found: {pipeline_args['config_path']}"
        pipeline_args["run_name"] = f"data_prep_run_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"
        logger.info("Starting data preparation pipeline")
        data_preparation_pipeline.with_options(**pipeline_args)()

    if run_upload_dataset:
        pipeline_args["config_path"] = str(root_dir / "configs" / "dataset_upload.yaml")
        assert pipeline_args["config_path"].exists(), f"Config file not found: {pipeline_args['config_path']}"
        pipeline_args["run_name"] = f"upload_dataset_run_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"

        assert settings.HF_TOKEN, "HuggingFace token required. Set HF_TOKEN in .env"
        logger.info("Starting dataset upload pipeline")
        dataset_upload_pipeline.with_options(**pipeline_args)()

    if run_generate_gmbl:
        config_path = root_dir / "configs" / "dataset_generation.yaml"
        assert config_path.exists(), f"Config file not found: {config_path}"
        pipeline_args["config_path"] = str(config_path)
        pipeline_args["run_name"] = f"generate_gmbl_run_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"

        logger.info("Starting GMBL dataset generation pipeline")
        
        dataset_generation_pipeline.with_options(**pipeline_args)()

    if run_render_diagram:
        logger.info("Starting diagram rendering script")
        subprocess.run([sys.executable, str(root_dir / "test_diagram.py")], check=True)

    if run_finetune:
        assert settings.HF_TOKEN, "HF_TOKEN required. Set it in .env file"
        assert settings.AWS_ARN_ROLE, "AWS_ARN_ROLE required. Set it in .env file"

        logger.info(f"Configuration: epochs={num_epochs}, batch_size={batch_size}, lr={learning_rate}")

        try:
            sagemaker_module = import_module("src.services.model.finetuning.sagemaker")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Missing SageMaker finetuning module: src.services.model.finetuning.sagemaker. "
                "Please restore this module or update --run-finetune to use the current training entrypoint."
            ) from exc

        run_finetuning_on_sagemaker = getattr(sagemaker_module, "run_finetuning_on_sagemaker", None)
        if run_finetuning_on_sagemaker is None:
            raise RuntimeError(
                "Function run_finetuning_on_sagemaker was not found in "
                "src.services.model.finetuning.sagemaker."
            )

        run_finetuning_on_sagemaker(
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            dataset_huggingface_workspace=dataset_workspace,
        )

        logger.info("Finetuning job submitted to AWS SageMaker successfully!")
        logger.info("Monitor progress in AWS SageMaker Console and Comet ML")

if __name__ == "__main__":
    main()
