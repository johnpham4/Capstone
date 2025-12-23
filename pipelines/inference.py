from zenml import pipeline
from loguru import logger

from steps.inference.test_inference import test_inference_step


@pipeline
def inference_pipeline(
    model_path: str,
    vq_model_path: str,
    test_prompt: str,
    output_dir: str
):
    """Inference pipeline to test trained model - generate image from text"""
    logger.info("Starting inference pipeline")

    result = test_inference_step(
        model_path=model_path,
        vq_model_path=vq_model_path,
        test_prompt=test_prompt,
        output_dir=output_dir
    )

    logger.success(f"Inference completed: {result}")

    return result
