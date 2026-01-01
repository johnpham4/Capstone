import re
import json
from loguru import logger
from abc import ABC, abstractmethod
from tqdm.auto import tqdm


from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from llm_engineering.applications.networks.qwen7B import QwenLocalLLM
from llm_engineering.applications.datasets.output_parser import ListPydanticOutputParser
from llm_engineering.domains.dataset import InstructDataset, InstructDatasetSample
from llm_engineering.domains.prompt import GenerateDatasetSamplesPrompt
from llm_engineering.applications.utils import misc

class DSLGenerator:
    def __init__(self):
        self.llm = QwenLocalLLM()
        self.parser = ListPydanticOutputParser(pydantic_object=InstructDatasetSample)

    @staticmethod
    def _extract_json_array(raw: str) -> str:
        """Extract JSON array from LLM output, handling markdown blocks and extra text."""
        # Remove markdown code blocks
        raw = re.sub(r'```json\s*', '', raw)
        raw = re.sub(r'```\s*', '', raw)

        # Find the last JSON array (most likely the actual output)
        # Look for pattern: Output JSON array with ONE object: [...]
        output_match = re.search(r'Output JSON array with ONE object:\s*(\[.*?\])(?:\s*$|\s*\])', raw, re.DOTALL)
        if output_match:
            return output_match.group(1)

        # Fallback: find any JSON array, prefer the last one
        matches = re.findall(r'(\[\s*\{.*?\}\s*\])', raw, re.DOTALL)
        if matches:
            return matches[-1]  # Return the last match

        # Last resort: try to find any array structure
        match = re.search(r'(\[.*\])', raw, re.DOTALL)
        if match:
            return match.group(1)

        raise ValueError(f"No JSON array found in LLM output: {raw[:200]}...")

    def __call__(self, system_prompt: str, prompts: list[GenerateDatasetSamplesPrompt]) -> InstructDataset:

        def _to_messages(prompt: GenerateDatasetSamplesPrompt) -> list[BaseMessage]:
            return [
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt.content)
            ]

        chain = self.llm | self.parser
        flattened_samples = []

        # Convert prompts to messages and batch
        string_prompts = [_to_messages(p) for p in prompts]
        batches = misc.batch(string_prompts, size=24)

        for batch in tqdm(batches, desc="Processing batches"):
            raw_results = chain.batch(batch)
            for raw in raw_results:
                try:
                    cleaned_json = self._extract_json_array(raw)
                    parsed_samples = self.parser.parse(cleaned_json)
                    flattened_samples.extend(parsed_samples)
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON decode error: {e}")
                    logger.debug(f"Raw output:\n{raw[:500]}...")
                except ValueError as e:
                    logger.warning(f"Extraction error: {e}")
                    logger.debug(f"Raw output:\n{raw[:500]}...")
                except Exception as e:
                    logger.exception(f"Unexpected error parsing LLM output: {e}")

        dataset = InstructDataset(samples=flattened_samples)
        logger.info(f"Generated {len(dataset.samples)} samples total (from {len(prompts)} prompts).")
        return dataset
