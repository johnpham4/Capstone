from zenml import step

from pipeline.services.unsloth_finetune.registry import registry_model


@step
def registry(
    num_train_epochs: int,
    per_device_train_batch_size: int,
    learning_rate: float,
    dataset_huggingface_workspace: str = "minn04",
    dataset_huggingface_repo_name: str = "text2dsl",
    is_dummy: bool = False,
) -> None:
    registry_model(
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        dataset_huggingface_workspace=dataset_huggingface_workspace,
        dataset_huggingface_repo_name=dataset_huggingface_repo_name,
        is_dummy=is_dummy,
    )
