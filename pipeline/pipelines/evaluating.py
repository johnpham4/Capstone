"""Evaluation pipeline — NOT YET IMPLEMENTED.

The `pipeline.steps.evaluating` module does not exist yet.
This file is a placeholder for future evaluation pipeline work.
"""

from zenml import pipeline


@pipeline
def evaluating(
    is_dummy: bool = False,
) -> None:
    # TODO: implement pipeline.steps.evaluating.evaluate step
    raise NotImplementedError("Evaluation pipeline steps not yet implemented")
