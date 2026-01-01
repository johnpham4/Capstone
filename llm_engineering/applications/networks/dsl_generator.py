import re
import json
from loguru import logger
from typing import List
from tqdm.auto import tqdm

from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from llm_engineering.applications.networks.qwen7B import QwenLocalLLM
from llm_engineering.applications.datasets.output_parser import ListPydanticOutputParser
from llm_engineering.domains.dataset import InstructDataset, InstructDatasetSample
from llm_engineering.domains.prompt import GenerateDatasetSamplesPrompt
from llm_engineering.applications.utils import misc


class DSLGenerator:
    """
    Generator for converting Vietnamese geometry problems to GMBL DSL format.

    Architecture:
    1. LLM generates raw JSON string
    2. Extract and clean JSON (handle malformed output)
    3. Parse to structured objects

    This separation allows better debugging when LLM output is unstable.
    """

    def __init__(self):
        self.llm = QwenLocalLLM()
        self.parser = ListPydanticOutputParser(pydantic_object=InstructDatasetSample)

    def _extract_json_array(self, raw: str) -> str:
        """
        Extract JSON array from LLM output.

        Handles common LLM output issues:
        - Markdown code blocks (```json)
        - Extra text before/after JSON
        - Trailing punctuation
        - Conversation markers (System:, Human:)

        Args:
            raw: Raw LLM output string

        Returns:
            Cleaned JSON string

        Raises:
            ValueError: If no valid JSON found
        """
        logger.debug(f"Raw output preview: {raw[:200]}")

        # Clean conversation markers
        raw = re.sub(r'^(System|Human|Assistant):\s*', '', raw, flags=re.MULTILINE)

        # Remove markdown
        raw = re.sub(r'```json\s*', '', raw)
        raw = re.sub(r'```\s*', '', raw)
        raw = raw.strip()

        # Pattern 1: Strict match with instruction + answer
        pattern = r'(\[\s*\{\s*"instruction".*?"answer".*?\}\s*\])'
        match = re.search(pattern, raw, re.DOTALL)
        if match:
            json_str = match.group(1)
            json_str = re.sub(r'[.,;!?]+\s*$', '', json_str)  # Clean trailing punctuation
            logger.debug(f"Extracted (pattern 1): {json_str[:100]}...")
            return json_str.strip()

        # Pattern 2: Relaxed match
        pattern = r'(\[.*?\{.*?"instruction".*?"answer".*?\}.*?\])'
        match = re.search(pattern, raw, re.DOTALL)
        if match:
            json_str = match.group(1)
            json_str = re.sub(r'[.,;!?]+\s*$', '', json_str)
            logger.debug(f"Extracted (pattern 2): {json_str[:100]}...")
            return json_str.strip()

        # Pattern 3: Entire output if it looks like JSON
        if raw.startswith('[') and raw.endswith(']'):
            logger.debug("Using full output as JSON")
            return raw

        raise ValueError(f"No valid JSON in output. Preview: {raw[:300]}")

    def _process_single_prompt(
        self,
        raw_output: str,
        prompt_num: int,
        total: int
    ) -> List[InstructDatasetSample]:
        """
        Process a single LLM output to extract dataset samples.

        Args:
            raw_output: Raw string from LLM
            prompt_num: Current prompt number (for logging)
            total: Total prompts (for logging)

        Returns:
            List of parsed InstructDatasetSample objects

        Raises:
            Various exceptions for logging purposes (caught by caller)
        """

        cleaned_json = self._extract_json_array(raw_output)

        samples = self.parser.parse(cleaned_json)

        logger.info(f"✓ Success - Generated {len(samples)} sample(s)")
        return samples

    def __call__(
        self,
        system_prompt: str,
        prompts: List[GenerateDatasetSamplesPrompt]
    ) -> InstructDataset:
        """
        Generate GMBL dataset from Vietnamese geometry problems.

        Args:
            system_prompt: System instruction for LLM
            prompts: List of geometry problems to convert

        Returns:
            InstructDataset with successfully generated samples
        """

        def _to_messages(prompt: GenerateDatasetSamplesPrompt) -> List[BaseMessage]:
            """Convert prompt to LangChain message format."""
            return [
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt.content)
            ]

        # Initialize tracking
        all_samples = []
        stats = {"success": 0, "failed": 0, "total": len(prompts)}

        # Prepare batches
        messages_batch = [_to_messages(p) for p in prompts]
        batches = list(misc.batch(messages_batch, size=4))

        logger.info(f"Processing {stats['total']} prompts in {len(batches)} batches...")

        # Process each batch
        for batch_idx, batch in enumerate(tqdm(batches, desc="Processing batches")):
            logger.info(f"Batch {batch_idx + 1}/{len(batches)} - {len(batch)} prompts")

            # Get raw outputs from LLM (no auto-parsing)
            raw_outputs = self.llm.batch(batch)

            # Process each output
            for idx, raw_output in enumerate(raw_outputs):
                prompt_num = batch_idx * 4 + idx + 1

                try:
                    # Process with detailed logging
                    samples = self._process_single_prompt(raw_output, prompt_num, stats['total'])
                    all_samples.extend(samples)
                    stats['success'] += 1

                except json.JSONDecodeError as e:
                    stats['failed'] += 1
                    logger.error(f"✗ JSON parse error: {e}")
                    logger.error(f"Raw output:\n{raw_output[:800]}\n")

                except ValueError as e:
                    stats['failed'] += 1
                    logger.error(f"✗ Extraction error: {e}")
                    logger.error(f"Raw output:\n{raw_output[:800]}\n")

                except Exception as e:
                    stats['failed'] += 1
                    logger.exception(f"✗ Unexpected error: {e}")
                    logger.error(f"Raw output:\n{raw_output[:800]}\n")

            # Batch summary
            logger.info(
                f"Batch {batch_idx + 1} done: "
                f"{stats['success']} success, {stats['failed']} failed"
            )

        # Final summary
        success_rate = (stats['success'] / stats['total'] * 100) if stats['total'] > 0 else 0
        logger.info(f"GENERATION COMPLETE")
        logger.info(f"Total prompts: {stats['total']}")
        logger.info(f"Successful: {stats['success']} ({success_rate:.1f}%)")
        logger.info(f"Failed: {stats['failed']} ({100-success_rate:.1f}%)")
        logger.info(f"Generated samples: {len(all_samples)}")

        return InstructDataset(samples=all_samples)
