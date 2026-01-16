from loguru import logger

from llm_engineering.model.inference.inference import LLMInferenceSagemakerEndpoint
from llm_engineering.model.inference.run import InferenceExecutor
from llm_engineering.settings import settings

if __name__ == "__main__":
    text = "Tam giác ABC, điểm D là trung điểm của đoạn thẳng AB, điểm E là trung điểm của đoạn thẳng AC, đường thẳng DE, BC song song với DE, đường thẳng là đường trung trực của đoạn thẳng BC"
    logger.info(f"Running inference for text: '{text}'")
    llm = LLMInferenceSagemakerEndpoint(
        endpoint_name=settings.SAGEMAKER_ENDPOINT_INFERENCE, inference_component_name=None
    )
    answer = InferenceExecutor(llm, text).execute()

    logger.info(f"Answer: '{answer}'")
    logger.info(f"Origin: (param (A B C))\n(define D point (midp A B))\n(define E point (midp A C))\n(param LDE line (through D))\n(param LBC line (through B))\n(assert (on-line C LBC))\n(assert (para LBC LDE))\n(param Lperp line (through (midp B C)))\n(assert (perp Lperp LBC))")
