<<<<<<<< HEAD:src/services/datasets/utils.py
from sklearn.model_selection import train_test_split

from src.models.domain.training import InstructDataset, InstructDatasetSample, InstructTrainTestSplit


def create_instruct_train_test_split(
    data: list[InstructDataset], test_size=0.2, random_state=42
) -> InstructTrainTestSplit:
    for dataset in data:
        samples = dataset.samples
        samples_dicts = [sample.model_dump() for sample in samples]

        if len(samples_dicts) > 0:
            # Allow disabling the test split by setting test_size to 0 or None.
            if test_size in (0, 0.0, None):
                train_samples_dicts = samples_dicts
                test_samples_dicts = []
            else:
                train_samples_dicts, test_samples_dicts = train_test_split(
                    samples_dicts, test_size=test_size, random_state=random_state
                )

            train_samples = [InstructDatasetSample(**sample_dict) for sample_dict in train_samples_dicts]
            test_samples = [InstructDatasetSample(**sample_dict) for sample_dict in test_samples_dicts]

        else:
            train_samples = []
            test_samples = []

        train_dataset = InstructDataset(samples=train_samples)
        test_dataset = InstructDataset(samples=test_samples)

    effective_test_size = 0.0 if test_size in (0, 0.0, None) else float(test_size)
    return InstructTrainTestSplit(train=train_dataset, test=test_dataset, test_split_size=effective_test_size)

========
from sklearn.model_selection import train_test_split

from pipeline.domain import InstructDataset, InstructDatasetSample, InstructTrainTestSplit


def create_instruct_train_test_split(
    data: list[InstructDataset], test_size=0.2, random_state=42
) -> InstructTrainTestSplit:
    for dataset in data:
        samples = dataset.samples
        samples_dicts = [sample.model_dump() for sample in samples]

        if len(samples_dicts) > 0:
            train_samples_dicts, test_samples_dicts = train_test_split(
                samples_dicts, test_size=test_size, random_state=random_state
            )

            train_samples = [InstructDatasetSample(**sample_dict) for sample_dict in train_samples_dicts]
            test_samples = [InstructDatasetSample(**sample_dict) for sample_dict in test_samples_dicts]

        else:
            train_samples = []
            test_samples = []

        train_dataset = InstructDataset(samples=train_samples)
        test_dataset = InstructDataset(samples=test_samples)

    return InstructTrainTestSplit(train=train_dataset, test=test_dataset, test_split_size=test_size)

>>>>>>>> minh-re:pipeline/services/datasets/utils.py
