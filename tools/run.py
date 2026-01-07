import click
from pathlib import Path
from loguru import logger
from datetime import datetime as dt
import yaml

from llm_engineering.domains.training_config import TrainingConfig
from llm_engineering.settings import settings


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
def main(
    no_cache: bool = False,
    run_prepare_data: bool = False,
    run_upload_dataset: bool = False,
    run_generate_gmbl: bool = False,
) -> None:
    assert run_prepare_data or run_upload_dataset or run_generate_gmbl, "Please use one of the options"

    pipeline_args = {"enable_cache": not no_cache}
    root_dir = Path(__file__).resolve().parent.parent

    # if run_extract:
    #     pipeline_args["config_path"] = root_dir / "configs" / "figure_extraction.yaml"
    #     assert pipeline_args["config_path"].exists(), f"Config file not found: {pipeline_args['config_path']}"
    #     pipeline_args["run_name"] = f"extract_pipeline_run_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"

    #     logger.info("Starting extraction pipeline (figures + text)")
    #     figure_extraction_pipeline.with_options(**pipeline_args)()

    if run_prepare_data:
        pipeline_args["config_path"] = root_dir / "configs" / "data_preparation.yaml"
        assert pipeline_args["config_path"].exists(), f"Config file not found: {pipeline_args['config_path']}"
        pipeline_args["run_name"] = f"data_prep_run_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"

        logger.info("Starting data preparation pipeline")
        from pipelines.data_preparation import data_preparation_pipeline
        data_preparation_pipeline.with_options(**pipeline_args)()

    if run_upload_dataset:
        pipeline_args["config_path"] = root_dir / "configs" / "dataset_upload.yaml"
        assert pipeline_args["config_path"].exists(), f"Config file not found: {pipeline_args['config_path']}"
        pipeline_args["run_name"] = f"upload_dataset_run_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"

        assert settings.HF_TOKEN, "HuggingFace token required. Set HF_TOKEN in .env"
        logger.info("Starting dataset upload pipeline")
        from pipelines.dataset_upload import dataset_upload_pipeline
        dataset_upload_pipeline.with_options(**pipeline_args)()

    if run_generate_gmbl:
        pipeline_args["config_path"] = root_dir / "configs" / "dataset_generation.yaml"
        assert pipeline_args["config_path"].exists(), f"Config file not found: {pipeline_args['config_path']}"
        pipeline_args["run_name"] = f"generate_gmbl_run_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"

        logger.info("Starting GMBL dataset generation pipeline")
        from pipelines.dataset_generation import dataset_generation_pipeline
        dataset_generation_pipeline.with_options(**pipeline_args)()

if __name__ == "__main__":
    main()