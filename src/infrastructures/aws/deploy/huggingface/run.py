from sagemaker.model import Model
from sagemaker.session import Session
from loguru import logger

from src.config.settings.base import settings

def create_endpoint():
    session = Session()

    logger.info(f"Creating vLLM endpoint: {settings.SAGEMAKER_ENDPOINT_INFERENCE}")
    logger.info(f"Model: {settings.HF_MODEL_ID}")

    model = Model(
        image_uri="637931482580.dkr.ecr.us-east-1.amazonaws.com/vllm:0.11.1",
        role=settings.AWS_ARN_ROLE,
        sagemaker_session=session,
        env={
            "SM_VLLM_MODEL": settings.HF_MODEL_ID,
            "HUGGING_FACE_HUB_TOKEN": settings.HF_TOKEN,
        },
    )

    model.deploy(
        instance_type=settings.GPU_INSTANCE_TYPE,  # ml.g5.2xlarge
        initial_instance_count=1,
        endpoint_name=settings.SAGEMAKER_ENDPOINT_INFERENCE,
        inference_ami_version="al2-ami-sagemaker-inference-gpu-3-1",
        wait=True,

    )

if __name__ == "__main__":
    create_endpoint()
