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
        # Remove any conversation markers (System:, Human:, Assistant:)
        raw = re.sub(r'^(System|Human|Assistant):\s*', '', raw, flags=re.MULTILINE)

        # Remove markdown code blocks
        raw = re.sub(r'```json\s*', '', raw)
        raw = re.sub(r'```\s*', '', raw)

        # Find JSON array - look for pattern with balanced brackets
        # Pattern 1: After "JSON output:" or similar
        output_match = re.search(r'(?:JSON output|Output):\s*(\[\s*\{.*?\}\s*\])', raw, re.DOTALL | re.IGNORECASE)
        if output_match:
            json_str = output_match.group(1)
            # Clean trailing punctuation
            json_str = re.sub(r'[.,;!?]+\s*$', '', json_str)
            return json_str.strip()

        # Pattern 2: Find the last complete JSON array
        matches = re.findall(r'(\[\s*\{[^\[\]]*"instruction"[^\[\]]*"answer"[^\[\]]*\}\s*\])', raw, re.DOTALL)
        if matches:
            json_str = matches[-1]
            json_str = re.sub(r'[.,;!?]+\s*$', '', json_str)
            return json_str.strip()

        # Pattern 3: Any JSON array structure
        match = re.search(r'(\[\s*\{.*?\}\s*\])', raw, re.DOTALL)
        if match:
            json_str = match.group(1)
            json_str = re.sub(r'[.,;!?]+\s*$', '', json_str)
            return json_str.strip()

        raise ValueError(f"No JSON array found in LLM output: {raw[:300]}...")

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
