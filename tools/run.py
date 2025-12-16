import click
from pathlib import Path
from loguru import logger
from datetime import datetime as dt

from llm_engineering.domains.training_config import TrainingConfig
from llm_engineering.settings import settings
from pipelines.figure_extraction import figure_extraction_pipeline
from pipelines.dataset_upload import dataset_upload_pipeline


@click.command()
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Disable caching for the pipeline run.",
)
@click.option(
    "--run-extract",
    is_flag=True,
    default=False,
    help="Run extraction pipeline (figures + text)"
)
@click.option(
    "--run-upload-dataset",
    is_flag=True,
    default=False,
    help="Upload dataset to HuggingFace Hub"
)
def main(
    no_cache: bool = False,
    run_extract: bool = False,
    run_upload_dataset: bool = False,
) -> None:
    assert run_extract or run_upload_dataset, "Please use one of the options"

    pipeline_args = {"enable_cache": not no_cache}
    root_dir = Path(__file__).resolve().parent.parent

    if run_extract:
        pipeline_args["config_path"] = root_dir / "configs" / "figure_extraction.yaml"
        assert pipeline_args["config_path"].exists(), f"Config file not found: {pipeline_args['config_path']}"
        pipeline_args["run_name"] = f"extract_pipeline_run_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"

        logger.info("Starting extraction pipeline (figures + text)")
        figure_extraction_pipeline.with_options(**pipeline_args)()

    if run_upload_dataset:
        pipeline_args["config_path"] = root_dir / "configs" / "dataset_upload.yaml"
        assert pipeline_args["config_path"].exists(), f"Config file not found: {pipeline_args['config_path']}"
        pipeline_args["run_name"] = f"upload_dataset_run_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"

        assert settings.HF_TOKEN, "HuggingFace token required. Set HF_TOKEN in .env"
        logger.info("Starting dataset upload pipeline")
        dataset_upload_pipeline.with_options(**pipeline_args)()


if __name__ == "__main__":
    main()