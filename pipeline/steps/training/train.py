from zenml import step

from pipeline.services.unsloth_finetune.sagemaker import run_finetuning_on_sagemaker


@step
def train(
    num_train_epochs: int,
    per_device_train_batch_size: int,
    learning_rate: float,
    dataset_huggingface_workspace: str = "minn04",
    dataset_huggingface_repo_name: str = "text2dsl",
    is_dummy: bool = False,
) -> None:
    run_finetuning_on_sagemaker(
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        dataset_huggingface_workspace=dataset_huggingface_workspace,
        dataset_huggingface_repo_name=dataset_huggingface_repo_name,
        is_dummy=is_dummy,
    )
