import json

from loguru import logger

try:
    from sagemaker.compute_resource_requirements.resource_requirements import ResourceRequirements
except ModuleNotFoundError:
    logger.warning("Couldn't load SageMaker imports. Run 'poetry install --with aws' to support AWS.")

from llm_src.settings import settings


hugging_face_deploy_config = {
    "SM_VLLM_MODEL": settings.HF_MODEL_ID,
    "SM_VLLM_HF_TOKEN": settings.HF_TOKEN,
    "SM_VLLM_TENSOR_PARALLEL_SIZE": str(settings.SM_NUM_GPUS),
    "SM_VLLM_MAX_MODEL_LEN": str(settings.MAX_TOTAL_TOKENS),
    "SM_VLLM_MAX_NUM_SEQS": str(settings.MAX_CONCURRENT_REQUESTS),
    "SM_VLLM_GPU_MEMORY_UTILIZATION": "0.9",
}

# hugging_face_deploy_config = {
#     "HF_MODEL_ID": settings.HF_MODEL_ID,
#     "HUGGING_FACE_HUB_TOKEN": settings.HF_TOKEN,
#     "SM_NUM_GPUS": json.dumps(settings.SM_NUM_GPUS),
#     "MAX_INPUT_LENGTH": json.dumps(settings.MAX_INPUT_LENGTH),
#     "MAX_TOTAL_TOKENS": json.dumps(settings.MAX_TOTAL_TOKENS),
#     "MAX_BATCH_TOTAL_TOKENS": json.dumps(settings.MAX_BATCH_TOTAL_TOKENS),
#     "MAX_BATCH_PREFILL_TOKENS": json.dumps(settings.MAX_BATCH_TOTAL_TOKENS),
#     "HF_MODEL_QUANTIZE": "bitsandbytes",
#     "USE_CACHE": json.dumps(settings.USE_CACHE_INFERENCE),
#     "MAX_CONCURRENT_REQUESTS": json.dumps(settings.MAX_CONCURRENT_REQUESTS),
#     "MAX_WAITING_TOKENS": json.dumps(settings.MAX_WAITING_TOKENS),
# }


model_resource_config = ResourceRequirements(
    requests={
        "copies": settings.COPIES,  # Number of replicas.
        "num_accelerators": settings.GPUS,  # Number of GPUs required.
        "num_cpus": settings.CPUS,  # Number of CPU cores required.
        "memory": 5 * 1024,  # Minimum memory required in Mb (required)
    },
)
