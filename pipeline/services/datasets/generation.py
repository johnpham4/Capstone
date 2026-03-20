from abc import ABC, abstractmethod
import time
from langchain_openai import ChatOpenAI
from loguru import logger
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.exceptions import OutputParserException

from src.config.settings.base import settings
from pipeline.domain import Document
from pipeline.domain import GenerateDatasetSamplesPrompt, Prompt
from pipeline.domain import InstructDataset, InstructDatasetSample, InstructTrainTestSplit
from pipeline.domain.prompt_dsl import prompt as DATASET_GENERATION_PROMPT

from . import utils as generation_utils

class DatasetGeneration(ABC):

    system_prompt_template = """You are a geometry formalization system.

You convert Vietnamese geometry problems into GMBL
(Geometry Meaning-Based Language), a formal geometry DSL.

STRICT RULES:
- Do NOT invent geometric objects
- Declare all objects before use
- Use correct predicate arity
- Follow the ontology provided by the user prompt
- Output ONLY valid JSON

Any violation is considered an error.
"""

    prompt_template_str: str | None = None

    @classmethod
    def get_system_prompt(cls) -> Prompt:
        return Prompt(
            template=cls.system_prompt_template,
            input_variables={},
            content=cls.system_prompt_template
        )

    @classmethod
    def get_prompt(cls, document: Document) -> GenerateDatasetSamplesPrompt:

        prompt_template = PromptTemplate.from_template(
            template=cls.prompt_template_str,
            template_format="jinja2",
        )

        input_variables = {"extract": document.caption_vn}

        prompt_text = prompt_template.format(**input_variables)

        return GenerateDatasetSamplesPrompt(
            template=prompt_template.template,
            input_variables=input_variables,
            content=prompt_text,
            document=document
        )

    @classmethod
    def generate(
        cls,
        prompts: list[GenerateDatasetSamplesPrompt],
        test_size: float = 0.2,
        batch_size: int = 4,
        sleep_seconds: float = 2.0,
        log_every_batches: int = 10,
    ) -> InstructTrainTestSplit:
        system_prompt_content = cls.get_system_prompt().content

        def _to_langchain(prompt: GenerateDatasetSamplesPrompt) -> list[BaseMessage]:
            return [
                SystemMessage(content=system_prompt_content),
                HumanMessage(content=prompt.content)
            ]

        assert settings.OPENAI_API_KEY is not None, "OpenAI API key must be set to generate datasets"

        llm = ChatOpenAI(
            model=settings.OPENAI_MODEL_ID,
            api_key=settings.OPENAI_API_KEY,
            max_tokens=512,  # Increased for complex GMBL
            temperature=0.3,  # Lower for more deterministic output
        )

        from langchain_core.output_parsers import JsonOutputParser
        parser = JsonOutputParser()
        chain = llm | parser

        total_batches = (len(prompts) + batch_size - 1) // batch_size if prompts else 0

        samples = []
        total_start = time.perf_counter()
        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = start + batch_size
            batch_prompts = prompts[start:end]
            batch = [_to_langchain(p) for p in batch_prompts]

            # Add delay between batches to avoid rate limit
            if batch_idx > 0 and sleep_seconds > 0:
                time.sleep(sleep_seconds)

            try:
                llm_start = time.perf_counter()
                raw_outputs = chain.batch(batch, stop=None)
                llm_elapsed = time.perf_counter() - llm_start
                post_process_start = time.perf_counter()

                if len(raw_outputs) != len(batch_prompts):
                    logger.warning(
                        f"Output/prompt count mismatch in batch {batch_idx + 1}: "
                        f"outputs={len(raw_outputs)} prompts={len(batch_prompts)}"
                    )

                for prompt, raw_output in zip(batch_prompts, raw_outputs):

                    # raw_output is either a dict or list
                    if isinstance(raw_output, list):
                        sample_dicts = raw_output
                    elif isinstance(raw_output, dict):
                        sample_dicts = [raw_output]
                    else:
                        logger.warning(f"Unexpected output type: {type(raw_output)}")
                        continue

                    # Inject image_dir into each dict BEFORE Pydantic validation
                    for sample_dict in sample_dicts:
                        sample_dict["image_dir"] = prompt.document.image_dir

                        # Now convert to Pydantic model
                        if isinstance(sample_dict.get("answer"), list):
                            sample_dict["answer"] = "\n".join(sample_dict["answer"])

                        try:
                            sample = InstructDatasetSample(**sample_dict)
                            samples.append(sample)
                        except Exception as e:
                            logger.error(f"Pydantic validation error: {e}")
                            logger.debug(f"Sample dict: {sample_dict}")

                post_process_elapsed = time.perf_counter() - post_process_start

                if (batch_idx + 1) % max(log_every_batches, 1) == 0 or batch_idx == 0:
                    done_batches = batch_idx + 1
                    elapsed_total = time.perf_counter() - total_start
                    avg_batch_seconds = elapsed_total / done_batches
                    remaining_batches = total_batches - done_batches
                    eta_minutes = (avg_batch_seconds * remaining_batches) / 60
                    logger.info(
                        (
                            f"Batch {done_batches}/{total_batches} completed | "
                            f"LLM: {llm_elapsed:.2f}s | "
                            f"Post-process: {post_process_elapsed:.2f}s | "
                            f"Avg/batch: {avg_batch_seconds:.2f}s | "
                            f"ETA: {eta_minutes:.1f}m | "
                            f"Samples so far: {len(samples)}"
                        )
                    )

            except OutputParserException as e:
                logger.error(f"Parse error in batch {batch_idx}: {str(e)}")
                logger.debug(f"Problematic output preview: {str(e)[:500]}")
            except Exception as e:
                logger.error(f"Unexpected error in batch {batch_idx}: {type(e).__name__}: {str(e)}")

        dataset = InstructDataset(samples=samples)
        logger.info(f"Generated {len(dataset.samples)} samples.")

        processed_datasets = cls.post_process_datasets(dataset, test_size=test_size)

        return processed_datasets

    @classmethod
    @abstractmethod
    def post_process_datasets(cls, dataset: InstructDataset, test_size: float) -> InstructTrainTestSplit:
        pass


class InstructiveDatasetGenerator(DatasetGeneration):
    prompt_template_str = DATASET_GENERATION_PROMPT

    @classmethod
    def post_process_datasets(cls, dataset: InstructDataset, test_size: float) -> InstructTrainTestSplit:

        return generation_utils.create_instruct_train_test_split([dataset], test_size=test_size, random_state=42)