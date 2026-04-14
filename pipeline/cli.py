import click
from pathlib import Path
from loguru import logger
from datetime import datetime as dt
from src.config.settings.settings import settings


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
    "--run-generate-questions",
    is_flag=True,
    default=False,
    help="Generate question dataset from Vietnamese captions"
)
@click.option(
    "--run-finetune",
    is_flag=True,
    default=False,
    help="Run finetuning on AWS SageMaker"
)
@click.option(
    "--run-finetune-peft-acemath",
    is_flag=True,
    default=False,
    help="Run PEFT finetuning for AceMath-1.5B on AWS SageMaker"
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
    default="quangne",
    help="Hugging Face workspace containing the dataset"
)
@click.option(
    "--dataset-repo",
    type=str,
    default="geometry3k8-8-1-1",
    help="Hugging Face dataset repository name"
)
@click.option(
    "--is-dummy",
    is_flag=True,
    default=False,
    help="Use a small subset of dataset for smoke finetuning"
)
@click.option(
    "--dummy-train-samples",
    type=int,
    default=400,
    show_default=True,
    help="Number of train samples when --is-dummy is enabled"
)
@click.option(
    "--dummy-eval-samples",
    type=int,
    default=100,
    show_default=True,
    help="Number of eval samples when --is-dummy is enabled"
)
def main(
    no_cache: bool = False,
    run_prepare_data: bool = False,
    run_upload_dataset: bool = False,
    run_generate_gmbl: bool = False,
    run_generate_questions: bool = False,
    run_finetune: bool = False,
    run_finetune_peft_acemath: bool = False,
    num_epochs: int = 1,
    batch_size: int = 2,
    learning_rate: float = 2e-4,
    dataset_workspace: str = "quangne",
    dataset_repo: str = "geometry3k8-8-1-1",
    is_dummy: bool = False,
    dummy_train_samples: int = 400,
    dummy_eval_samples: int = 100,
) -> None:
    assert (
        run_prepare_data
        or run_upload_dataset
        or run_generate_gmbl
        or run_generate_questions
        or run_finetune
        or run_finetune_peft_acemath
    ), "Please use one of the options"

    pipeline_args = {"enable_cache": not no_cache}
    pipeline_dir = Path(__file__).resolve().parent

    if run_prepare_data:
        pipeline_args["config_path"] = pipeline_dir / "configs" / "data_preparation.yaml"
        assert pipeline_args["config_path"].exists(), f"Config file not found: {pipeline_args['config_path']}"
        pipeline_args["run_name"] = f"data_prep_run_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"

        logger.info("Starting data preparation pipeline")
        from pipeline.pipelines.data_preparation import data_preparation_pipeline
        data_preparation_pipeline.with_options(**pipeline_args)()

    if run_upload_dataset:
        pipeline_args["config_path"] = pipeline_dir / "configs" / "dataset_upload.yaml"
        assert pipeline_args["config_path"].exists(), f"Config file not found: {pipeline_args['config_path']}"
        pipeline_args["run_name"] = f"upload_dataset_run_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"

        assert settings.HF_TOKEN, "HuggingFace token required. Set HF_TOKEN in .env"
        logger.info("Starting dataset upload pipeline")
        from pipeline.pipelines.dataset_upload import dataset_upload_pipeline
        dataset_upload_pipeline.with_options(**pipeline_args)()

    if run_generate_gmbl:
        pipeline_args["config_path"] = pipeline_dir / "configs" / "dataset_generation.yaml"
        assert pipeline_args["config_path"].exists(), f"Config file not found: {pipeline_args['config_path']}"
        pipeline_args["run_name"] = f"generate_gmbl_run_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"

        logger.info("Starting GMBL dataset generation pipeline")
        from pipeline.pipelines.dataset_generation import dataset_generation_pipeline
        dataset_generation_pipeline.with_options(**pipeline_args)()

    if run_generate_questions:
        pipeline_args["config_path"] = pipeline_dir / "configs" / "question_generation.yaml"
        assert pipeline_args["config_path"].exists(), f"Config file not found: {pipeline_args['config_path']}"
        pipeline_args["run_name"] = f"generate_questions_run_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"

        logger.info("Starting question generation pipeline")
        from pipeline.pipelines.question_generation import question_generation_pipeline
        question_generation_pipeline.with_options(**pipeline_args)()

    if run_finetune:
        assert settings.HF_TOKEN, "HF_TOKEN required. Set it in .env file"
        assert settings.AWS_ARN_ROLE, "AWS_ARN_ROLE required. Set it in .env file"

        logger.info(f"Configuration: epochs={num_epochs}, batch_size={batch_size}, lr={learning_rate}")

        from pipeline.services.unsloth_finetune.sagemaker import run_finetuning_on_sagemaker

        run_finetuning_on_sagemaker(
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            dataset_huggingface_workspace=dataset_workspace,
        )

        logger.info("Finetuning job submitted to AWS SageMaker successfully!")
        logger.info("Monitor progress in AWS SageMaker Console and Comet ML")

    if run_finetune_peft_acemath:
        assert settings.HF_TOKEN, "HF_TOKEN required. Set it in .env file"
        assert settings.AWS_ARN_ROLE, "AWS_ARN_ROLE required. Set it in .env file"

        logger.info(
            f"PEFT AceMath configuration: epochs={num_epochs}, batch_size={batch_size}, lr={learning_rate}, dataset={dataset_workspace}/{dataset_repo}, is_dummy={is_dummy}, dummy_train={dummy_train_samples}, dummy_eval={dummy_eval_samples}"
        )

        from pipeline.services.peft_finetuning.sagemaker import run_finetuning_on_sagemaker as run_peft_acemath

        run_peft_acemath(
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            dataset_huggingface_workspace=dataset_workspace,
            dataset_huggingface_repo_name=dataset_repo,
            model_output_huggingface_workspace="quangne",
            model_name="nvidia/AceMath-1.5B-Instruct",
            is_dummy=is_dummy,
            dummy_train_samples=dummy_train_samples,
            dummy_eval_samples=dummy_eval_samples,
        )

        logger.info("PEFT AceMath finetuning job submitted to AWS SageMaker successfully!")
        logger.info("Monitor progress in AWS SageMaker Console and Comet ML")

if __name__ == "__main__":
    main()