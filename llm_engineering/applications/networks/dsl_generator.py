import re
import json
from pathlib import Path
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

    def __init__(self, checkpoint_dir: str = "/tmp/dsl_checkpoints"):
        self.llm = QwenLocalLLM()
        self.parser = ListPydanticOutputParser(pydantic_object=InstructDatasetSample)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _save_checkpoint(self, samples: List[InstructDatasetSample], batch_idx: int, run_id: str):
        """Save intermediate results."""
        checkpoint_file = self.checkpoint_dir / f"{run_id}_batch_{batch_idx}.json"
        data = [{"instruction": s.instruction, "answer": s.answer} for s in samples]
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load_checkpoints(self, run_id: str) -> List[InstructDatasetSample]:
        """Load existing checkpoints."""
        samples = []
        for cp_file in sorted(self.checkpoint_dir.glob(f"{run_id}_batch_*.json")):
            with open(cp_file, 'r', encoding='utf-8') as f:
                for item in json.load(f):
                    samples.append(InstructDatasetSample(instruction=item["instruction"], answer=item["answer"]))
        if samples:
            logger.info(f"Resumed {len(samples)} samples from checkpoints")
        return samples

    def _extract_json_array(self, raw: str) -> str:
        """
        Extract valid JSON array from potentially malformed LLM output.

        Handles:
        - Markdown code blocks
        - Conversation markers (System:, Human:, Assistant:)
        - Trailing punctuation
        - Multiple extraction patterns

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

        # Parse to objects - parser might return single object or list
        parsed = self.parser.parse(cleaned_json)

        # Ensure result is always a list
        if isinstance(parsed, list):
            samples = parsed
        else:
            samples = [parsed]

        logger.info(f"Success - Generated {len(samples)} sample(s)")
        return samples

    def __call__(
        self,
        system_prompt: str,
        prompts: List[GenerateDatasetSamplesPrompt],
        run_id: str = "default",
        checkpoint_every: int = 10
    ) -> InstructDataset:
        """
        Generate GMBL dataset from Vietnamese geometry problems.

        Args:
            system_prompt: System instruction for LLM
            prompts: List of geometry problems to convert
            run_id: Unique ID for checkpointing
            checkpoint_every: Save checkpoint every N batches

        Returns:
            InstructDataset with successfully generated samples
        """

        # Try resume from checkpoint
        all_samples = self._load_checkpoints(run_id)
        start_batch = len(all_samples) // 16

        def _to_messages(prompt: GenerateDatasetSamplesPrompt) -> List[BaseMessage]:
            """Convert prompt to LangChain message format."""
            return [
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt.content)
            ]

        # Prepare batches
        messages_batch = [_to_messages(p) for p in prompts]
        batches = list(misc.batch(messages_batch, size=16))

        logger.info(f"Processing {len(prompts)} prompts in {len(batches)} batches...")
        if start_batch > 0:
            logger.info(f"Resuming from batch {start_batch + 1}")

        batch_samples = []

        # Process each batch
        for batch_idx in tqdm(range(start_batch, len(batches)), desc="Processing batches"):
            batch = batches[batch_idx]
            logger.info(f"Batch {batch_idx + 1}/{len(batches)}")

            # Get raw outputs from LLM (no auto-parsing)
            raw_outputs = self.llm.batch(batch)

            # Process each output
            for idx, raw_output in enumerate(raw_outputs):
                try:
                    samples = self._process_single_prompt(raw_output)
                    batch_samples.extend(samples)
                    all_samples.extend(samples)

                except json.JSONDecodeError as e:
                    logger.error(f"JSON parse error: {e}")

                except ValueError as e:
                    logger.error(f"Extraction error: {e}")

                except Exception as e:
                    logger.error(f"Unexpected error: {e}")

            # Checkpoint
            if (batch_idx + 1) % checkpoint_every == 0:
                self._save_checkpoint(batch_samples, batch_idx, run_id)
                batch_samples = []

        # Final checkpoint
        if batch_samples:
            self._save_checkpoint(batch_samples, len(batches) - 1, run_id)

        logger.info(f"Generated samples: {len(all_samples)}")

        return InstructDataset(samples=all_samples)
