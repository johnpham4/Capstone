import json

from loguru import logger

try:
    from sagemaker.compute_resource_requirements.resource_requirements import ResourceRequirements
except ModuleNotFoundError:
    logger.warning("Couldn't load SageMaker imports. Run 'poetry install --with aws' to support AWS.")

from llm_src.settings import settings

hugging_face_deploy_config = {
    "HF_MODEL_ID": settings.HF_MODEL_ID,
    "HUGGING_FACE_HUB_TOKEN": settings.HF_TOKEN,
    "SM_NUM_GPUS": json.dumps(settings.SM_NUM_GPUS),  # Number of GPU used per replica
    "MAX_INPUT_LENGTH": json.dumps(settings.MAX_INPUT_LENGTH),  # Max length of input text
    "MAX_TOTAL_TOKENS": json.dumps(settings.MAX_TOTAL_TOKENS),  # Max length of the generation (including input text)
    "MAX_BATCH_TOTAL_TOKENS": json.dumps(settings.MAX_BATCH_TOTAL_TOKENS),
    "MAX_BATCH_PREFILL_TOKENS": json.dumps(settings.MAX_BATCH_TOTAL_TOKENS),
    "HF_MODEL_QUANTIZE": "bitsandbytes",
    "USE_CACHE": json.dumps(settings.USE_CACHE_INFERENCE),  # Enable KV cache for autoregressive generation
    "MAX_CONCURRENT_REQUESTS": json.dumps(settings.MAX_CONCURRENT_REQUESTS),  # Concurrent request handling
    "MAX_WAITING_TOKENS": json.dumps(settings.MAX_WAITING_TOKENS),  # Token queue management
}


model_resource_config = ResourceRequirements(
    requests={
        "copies": settings.COPIES,  # Number of replicas.
        "num_accelerators": settings.GPUS,  # Number of GPUs required.
        "num_cpus": settings.CPUS,  # Number of CPU cores required.
        "memory": 5 * 1024,  # Minimum memory required in Mb (required)
    },
)
