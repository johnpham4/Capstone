from zenml import pipeline
from loguru import logger

from steps.inference.test_inference import test_inference_step


@pipeline
def inference_pipeline(
    model_path: str,
    vq_model_path: str,
    test_image_path: str,
    output_dir: str
):
    logger.info("Starting inference pipeline")

    result = test_inference_step(
        model_path=model_path,
        vq_model_path=vq_model_path,
        test_image_path=test_image_path,
        output_dir=output_dir
    )

    logger.success(f"Inference completed: {result}")

    return result
