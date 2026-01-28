from loguru import logger

try:
    from sagemaker.enums import EndpointType
    # from sagemaker.huggingface import get_huggingface_llm_image_uri
except ModuleNotFoundError:
    logger.warning("Couldn't load SageMaker imports. Run 'poetry install --with aws' to support AWS.")

from utils import ResourceManager
from llm_src.settings import settings

from config import hugging_face_deploy_config, model_resource_config
from sagemaker_huggingface import DeploymentService, SagemakerHuggingfaceStrategy

# AWS vLLM Documentation:
# - Example deployment: https://github.com/aws/deep-learning-containers/blob/master/test/vllm/sagemaker/test_sm_endpoint.py

# EndpointType.INFERENCE_MODEL_TYPE
def create_endpoint(endpoint_type=EndpointType.MODEL_BASED) -> None:
    assert settings.AWS_ARN_ROLE is not None, "AWS_ARN_ROLE is not set in the .env file."
    assert settings.HF_TOKEN is not None, "HF_TOKEN is not set in the .env file."

    logger.info(f"Creating vLLM endpoint: {settings.SAGEMAKER_ENDPOINT_INFERENCE}")
    logger.info(f"Model: {settings.HF_MODEL_ID}, Instance: {settings.GPU_INSTANCE_TYPE}")

    # vLLM image from AWS Deep Learning Containers
    # See: https://github.com/aws/deep-learning-containers/blob/master/available_images.md
    llm_image = "637931482580.dkr.ecr.us-east-1.amazonaws.com/vllm:0.11.1"

    resource_manager = ResourceManager()
    deployment_service = DeploymentService(resource_manager=resource_manager)

    SagemakerHuggingfaceStrategy(deployment_service).deploy(
        role_arn=settings.AWS_ARN_ROLE,
        llm_image=llm_image,
        config=hugging_face_deploy_config,
        endpoint_name=settings.SAGEMAKER_ENDPOINT_INFERENCE,
        endpoint_config_name=settings.SAGEMAKER_ENDPOINT_CONFIG_INFERENCE,
        gpu_instance_type=settings.GPU_INSTANCE_TYPE,
        resources=model_resource_config,
        endpoint_type=endpoint_type,
    )

    # Old code with HuggingFace TGI image (without vLLM)
    # llm_image = get_huggingface_llm_image_uri("huggingface", version="2.2.0")
    # OR
    # llm_image = "763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.2.0-tgi2.2.0-gpu-py310-cu121-ubuntu22.04"


if __name__ == "__main__":
    create_endpoint(endpoint_type=EndpointType.MODEL_BASED)
