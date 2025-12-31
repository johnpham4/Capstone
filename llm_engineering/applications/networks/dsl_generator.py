import re
import json
from loguru import logger
from typing import List
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.exceptions import OutputParserException

from llm_engineering.domains.dataset import InstructDataset, InstructDatasetSample
from llm_engineering.domains.prompt import GenerateDatasetSamplesPrompt
from llm_engineering.applications.networks.qwen7B import QwenLocalLLM
from llm_engineering.applications.utils import misc
from llm_engineering.applications.datasets.output_parser import ListPydanticOutputParser
from llm_engineering.applications.networks.base import SingletonMeta


class DSLGenerator(metaclass=SingletonMeta):
    """DSL Generator using QwenLocalLLM, safely parsing JSON output."""

    def __init__(self):
        self.llm = QwenLocalLLM()
        self.parser = ListPydanticOutputParser(pydantic_object=InstructDatasetSample)

    def __call__(self, system_prompt: str, prompts: List[GenerateDatasetSamplesPrompt]) -> InstructDataset:

        def _to_prompt(prompt: GenerateDatasetSamplesPrompt) -> str:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt.content),
            ]
            # Qwen expects single string
            return "\n".join(m.content for m in messages)

        string_prompts = [_to_prompt(p) for p in prompts]
        batches = misc.batch(string_prompts, size=12)  # nhỏ hơn nếu model 7B

        all_samples = []

        for batch in batches:
            try:
                outputs = [self.llm._call(p) for p in batch]

                for out in outputs:
                    # Extract JSON array from text (allow extra text before/after)
                    match = re.search(r"(\[.*\])", out, flags=re.DOTALL)
                    if not match:
                        logger.warning(f"No JSON array found in LLM output:\n{out}")
                        continue

                    json_str = match.group(1)

                    # Escape newlines if needed (some LLM output raw \n)
                    json_str = json_str.replace("\n", "\\n")

                    try:
                        json_array = json.loads(json_str)
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON decode error: {e}\nRaw output:\n{json_str}")
                        continue

                    # Parse each object into InstructDatasetSample
                    samples = self.parser._parse_obj(json_array)
                    all_samples.extend(samples)

            except Exception:
                logger.exception("Failed to parse LLM batch output")

        dataset = InstructDataset(samples=all_samples)
        logger.info(f"Generated {len(dataset.samples)} samples total.")
        return dataset
