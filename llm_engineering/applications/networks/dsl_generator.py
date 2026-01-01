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
        logger.debug(f"Raw LLM output (first 200 chars): {raw[:200]}")

        # Remove any conversation markers (System:, Human:, Assistant:)
        raw = re.sub(r'^(System|Human|Assistant):\s*', '', raw, flags=re.MULTILINE)

        # Remove markdown code blocks
        raw = re.sub(r'```json\s*', '', raw)
        raw = re.sub(r'```\s*', '', raw)

        # Clean up and normalize
        raw = raw.strip()

        # Pattern 1: Find JSON starting with [ and containing "instruction" and "answer"
        # Use non-greedy match and ensure we capture complete JSON
        json_pattern = r'(\[\s*\{\s*"instruction".*?"answer".*?\}\s*\])'
        match = re.search(json_pattern, raw, re.DOTALL)
        if match:
            json_str = match.group(1)
            # Clean trailing punctuation but preserve JSON structure
            json_str = re.sub(r'[.,;!?]+\s*$', '', json_str)
            logger.debug(f"Extracted JSON: {json_str[:100]}...")
            return json_str.strip()

        # Pattern 2: Try to find any valid JSON array
        match = re.search(r'(\[.*?\{.*?"instruction".*?"answer".*?\}.*?\])', raw, re.DOTALL)
        if match:
            json_str = match.group(1)
            json_str = re.sub(r'[.,;!?]+\s*$', '', json_str)
            logger.debug(f"Extracted JSON (pattern 2): {json_str[:100]}...")
            return json_str.strip()

        # Pattern 3: Last resort - look for anything that looks like JSON
        if raw.startswith('[') and raw.endswith(']'):
            logger.debug("Using entire output as JSON")
            return raw.strip()

        raise ValueError(f"No valid JSON array found in output. Raw (300 chars): {raw[:300]}")

    def __call__(self, system_prompt: str, prompts: list[GenerateDatasetSamplesPrompt]) -> InstructDataset:

        def _to_messages(prompt: GenerateDatasetSamplesPrompt) -> list[BaseMessage]:
            return [
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt.content)
            ]

        chain = self.llm | self.parser
        flattened_samples = []

        total_prompts = len(prompts)
        successful_count = 0
        failed_count = 0

        # Convert prompts to messages and batch
        string_prompts = [_to_messages(p) for p in prompts]
        batches = list(misc.batch(string_prompts, size=4))  # Smaller batch for debugging

        logger.info(f"Processing {total_prompts} prompts in {len(batches)} batches...")

        for batch_idx, batch in enumerate(tqdm(batches, desc="Processing batches")):
            logger.info(f"\n{'='*60}")
            logger.info(f"Batch {batch_idx + 1}/{len(batches)} - Processing {len(batch)} prompts")
            logger.info(f"{'='*60}")

            raw_results = chain.batch(batch)

            for idx, raw in enumerate(raw_results):
                prompt_num = batch_idx * 4 + idx + 1
                logger.info(f"\n--- Prompt {prompt_num}/{total_prompts} ---")

                try:
                    logger.debug(f"Raw output length: {len(raw)} chars")
                    cleaned_json = self._extract_json_array(raw)
                    parsed_samples = self.parser.parse(cleaned_json)
                    flattened_samples.extend(parsed_samples)
                    successful_count += 1
                    logger.info(f"✓ Success - Generated {len(parsed_samples)} sample(s)")

                except json.JSONDecodeError as e:
                    failed_count += 1
                    logger.error(f"✗ JSON decode error: {e}")
                    logger.error(f"Raw output:\n{raw[:800]}")

                except ValueError as e:
                    failed_count += 1
                    logger.error(f"✗ Extraction error: {e}")
                    logger.error(f"Raw output:\n{raw[:800]}")

                except Exception as e:
                    failed_count += 1
                    logger.exception(f"✗ Unexpected error: {e}")
                    logger.error(f"Raw output:\n{raw[:800]}")

            # Progress summary after each batch
            logger.info(f"\nBatch {batch_idx + 1} complete: {successful_count} success, {failed_count} failed")

        dataset = InstructDataset(samples=flattened_samples)
        logger.info(f"\n{'='*60}")
        logger.info(f"FINAL RESULTS:")
        logger.info(f"Total prompts: {total_prompts}")
        logger.info(f"Successful: {successful_count} ({successful_count/total_prompts*100:.1f}%)")
        logger.info(f"Failed: {failed_count} ({failed_count/total_prompts*100:.1f}%)")
        logger.info(f"Generated samples: {len(dataset.samples)}")
        logger.info(f"{'='*60}")
        return dataset
