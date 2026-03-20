<<<<<<< HEAD:pipelines/training.py
from zenml import pipeline

from steps import training as training_steps


@pipeline
def training(
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 2,
    learning_rate: float = 3e-4,
    dataset_huggingface_workspace: str = "minn04",
    dataset_huggingface_repo_name: str = "text2dsl",
    is_dummy: bool = False,
) -> None:
    training_steps.train(
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        dataset_huggingface_workspace=dataset_huggingface_workspace,
        dataset_huggingface_repo_name=dataset_huggingface_repo_name,
        is_dummy=is_dummy,
    )
=======
from zenml import pipeline

from pipeline.steps import training as training_steps


@pipeline
def training(
    model_name: str,
    prompt: str,
    version: str,
    alias: str = "",
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 2,
    learning_rate: float = 3e-4,
    dataset_huggingface_workspace: str = "minn04",
    dataset_huggingface_repo_name: str = "text2dsl",
    is_dummy: bool = False,

) -> None:
    training_steps.train(
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        dataset_huggingface_workspace=dataset_huggingface_workspace,
        dataset_huggingface_repo_name=dataset_huggingface_repo_name,
        is_dummy=is_dummy,
    )
>>>>>>> 6cf03dda8dad8bb8fa1226b8b4e9166c3f287527:pipeline/pipelines/training.py
