import re
import json
from loguru import logger
from typing import List
from tqdm.auto import tqdm

from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage

from llm_engineering.applications.networks.base import SingletonMeta
from llm_engineering.applications.networks.qwen7B import QwenLocalLLM
from llm_engineering.applications.datasets.output_parser import ListPydanticOutputParser
from llm_engineering.domains.dataset import InstructDataset, InstructDatasetSample
from llm_engineering.domains.prompt import GenerateDatasetSamplesPrompt
from llm_engineering.applications.utils import misc


class DSLGenerator(metaclass=SingletonMeta):
    def __init__(self):
        self.llm = QwenLocalLLM()
        self.parser = ListPydanticOutputParser(pydantic_object=InstructDatasetSample)

    def _extract_json_array(self, raw: str) -> str:
        raw = re.sub(r'^(System|Human|Assistant):\s*', '', raw, flags=re.MULTILINE)
        raw = re.sub(r'```json\s*', '', raw)
        raw = re.sub(r'```\s*', '', raw)
        raw = raw.strip()

        pattern = r'(\[\s*\{\s*"instruction".*?"answer".*?\}\s*\])'
        match = re.search(pattern, raw, re.DOTALL)
        if match:
            json_str = match.group(1)
            json_str = re.sub(r'[.,;!?]+\s*$', '', json_str)
            logger.debug(f"Extracted (pattern 1): {json_str[:100]}...")
            return json_str.strip()

        pattern = r'(\[.*?\{.*?"instruction".*?"answer".*?\}.*?\])'
        match = re.search(pattern, raw, re.DOTALL)
        if match:
            json_str = match.group(1)
            json_str = re.sub(r'[.,;!?]+\s*$', '', json_str)
            logger.debug(f"Extracted (pattern 2): {json_str[:100]}...")
            return json_str.strip()

        if raw.startswith('[') and raw.endswith(']'):
            logger.debug("Using full output as JSON")
            return raw

        raise ValueError(f"No valid JSON in output. Preview: {raw[:300]}")

    def _process_single_prompt(self, raw_output: str, image_dir: str) -> List[InstructDatasetSample]:
        """Parse LLM output and inject image_dir into each sample"""
        cleaned_json = self._extract_json_array(raw_output)

        # Parse as raw dict first (not Pydantic yet)
        raw_data = json.loads(cleaned_json)

        if not isinstance(raw_data, list):
            raw_data = [raw_data]

        # Inject image_dir into each dict before Pydantic validation
        for item in raw_data:
            item['image_dir'] = image_dir

        # Now convert to Pydantic models
        samples = [InstructDatasetSample(**item) for item in raw_data]

        logger.info(f"Success - Generated {len(samples)} sample(s)")
        return samples

    def __call__(
        self,
        system_prompt: str,
        prompts: List[GenerateDatasetSamplesPrompt],
        batch_size: int = 16
    ) -> InstructDataset:
        all_samples = []

        def _to_messages(prompt: GenerateDatasetSamplesPrompt) -> List[BaseMessage]:
            return [
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt.content)
            ]

        messages_batch = [_to_messages(p) for p in prompts]
        batches = list(misc.batch(messages_batch, size=batch_size))

        logger.info(f"Processing {len(prompts)} prompts in {len(batches)} batches...")

        for batch_idx in tqdm(range(len(batches)), desc="Processing batches"):
            batch = batches[batch_idx]
            logger.info(f"Batch {batch_idx + 1}/{len(batches)}")

            raw_outputs = self.llm.batch(batch)

            for idx, raw_output in enumerate(raw_outputs):
                # Get corresponding prompt to extract image_dir
                prompt_idx = batch_idx * batch_size + idx
                prompt = prompts[prompt_idx]

                try:
                    samples = self._process_single_prompt(raw_output, prompt.document.image_dir)
                    all_samples.extend(samples)

                except json.JSONDecodeError as e:
                    logger.error(f"JSON parse error: {e}")

                except ValueError as e:
                    logger.error(f"Extraction error: {e}")

                except Exception as e:
                    logger.error(f"Unexpected error: {e}")

        logger.info(f"Generated samples: {len(all_samples)}")

        return InstructDataset(samples=all_samples)
