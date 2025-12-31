from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from loguru import logger
import re
import json

from llm_engineering.applications.datasets.output_parser import ListPydanticOutputParser
from llm_engineering.applications.networks.base import SingletonMeta
from llm_engineering.domains.dataset import InstructDataset, InstructDatasetSample
from llm_engineering.domains.prompt import GenerateDatasetSamplesPrompt
from llm_engineering.applications.networks.qwen7B import QwenLocalLLM
from llm_engineering.applications.utils import misc

class DSLGenerator(metaclass=SingletonMeta):

    def __init__(self):
        self.llm = QwenLocalLLM()

    def __call__(
        self,
        system_prompt: str,
        prompts: list[GenerateDatasetSamplesPrompt],
    ) -> InstructDataset:

        def _to_prompt(prompt: GenerateDatasetSamplesPrompt) -> str:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt.content),
            ]
            return "\n".join(m.content for m in messages)

        parser = ListPydanticOutputParser(
            pydantic_object=InstructDatasetSample
        )

        chain = self.llm | parser

        string_prompts = [_to_prompt(p) for p in prompts]
        batches = misc.batch(string_prompts, size=24)

        flattened_samples = []

        # for batch in batches:
        #     try:
        #         results = chain.batch(batch)
        #         for r in results:
        #             flattened_samples.extend(r)
        #     except OutputParserException:
        #         logger.exception("Failed to parse output JSON")

        for batch in batches:
            raw_results = self.llm.batch(batch)  # raw string outputs
            for r in raw_results:
                try:
                    # Lọc JSON array từ text (bỏ text thừa)
                    json_array = re.search(r"(\[.*\])", r, flags=re.DOTALL)
                    if not json_array:
                        raise ValueError("No JSON array found in LLM output")
                    data = json.loads(json_array.group(1))
                    flattened_samples.extend(self.parser._parse_obj(data))
                except Exception:
                    logger.exception("Failed to parse output JSON")

        dataset = InstructDataset(samples=flattened_samples)
        logger.info(f"Generated {len(dataset.samples)} samples total.")
        return dataset
