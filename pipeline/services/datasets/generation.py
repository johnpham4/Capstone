from abc import ABC, abstractmethod
import re
import time
from langchain_openai import ChatOpenAI
from loguru import logger
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.exceptions import OutputParserException

from src.config.settings.settings import settings
from src.services.diagram.dsl_parser import DSLParser
from pipeline.domain import Document
from pipeline.domain import GenerateDatasetSamplesPrompt, Prompt
from pipeline.domain import InstructDataset, InstructDatasetSample, InstructTrainTestSplit
from pipeline.domain.prompt_dsl import prompt as DATASET_GENERATION_PROMPT
from pipeline.domain.prompt_question import prompt as QUESTION_GENERATION_PROMPT

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
        if cls.prompt_template_str is None:
            raise ValueError("prompt_template_str must be set before generating prompts")

        extract_text = str(document.caption_vn)
        input_variables = {"extract": extract_text}

        if "{{ extract }}" in cls.prompt_template_str:
            prompt_text = cls.prompt_template_str.replace("{{ extract }}", extract_text)
        elif "{extract}" in cls.prompt_template_str:
            prompt_text = cls.prompt_template_str.format(extract=extract_text)
        else:
            prompt_text = f"{cls.prompt_template_str}\n\n{extract_text}"

        return GenerateDatasetSamplesPrompt(
            template=cls.prompt_template_str,
            input_variables=input_variables,
            content=prompt_text,
            document=document
        )

    @classmethod
    def _normalize_and_validate_dsl(cls, answer: str) -> tuple[str, bool]:
        """Best-effort normalize DSL line parentheses and validate parseability."""
        if not isinstance(answer, str):
            return "", False

        raw_lines = [ln.strip() for ln in answer.splitlines() if ln.strip()]
        if not raw_lines:
            return "", False

        def _clean_prefix(line: str) -> str:
            line = line.strip().strip('"')
            if line.startswith("```"):
                return ""
            line = re.sub(r"^[-*•]\s+", "", line)
            line = re.sub(r"^\d+[\.)]\s+", "", line)
            line = re.sub(r"^→\s*", "", line)
            return line.strip()

        def _balance_line_parens(line: str) -> str:
            fixed = line
            if not fixed.startswith("("):
                fixed = "(" + fixed
            if not fixed.endswith(")"):
                fixed = fixed + ")"

            open_count = fixed.count("(")
            close_count = fixed.count(")")
            if close_count > open_count:
                extra = close_count - open_count
                while extra > 0 and fixed.endswith(")"):
                    fixed = fixed[:-1]
                    extra -= 1

            open_count = fixed.count("(")
            close_count = fixed.count(")")
            if open_count > close_count:
                fixed = fixed + (")" * (open_count - close_count))

            return fixed

        normalized_lines: list[str] = []
        for line in raw_lines:
            cleaned = _clean_prefix(line)
            if not cleaned:
                continue
            normalized_lines.append(_balance_line_parens(cleaned))

        if not normalized_lines:
            return "", False

        joined = "\n".join(normalized_lines)
        open_count = joined.count("(")
        close_count = joined.count(")")
        if close_count > open_count:
            extra = close_count - open_count
            while extra > 0 and joined.endswith(")"):
                joined = joined[:-1]
                extra -= 1
            if joined.count(")") > joined.count("("):
                return joined, False
        if open_count > close_count:
            joined = joined + (")" * (open_count - close_count))

        normalized_lines = [ln for ln in joined.splitlines() if ln.strip()]

        parser = DSLParser()
        try:
            for line in normalized_lines:
                parsed = parser.parse_sexpr(line)
                if parsed is None:
                    return "\n".join(normalized_lines), False
        except Exception:
            return "\n".join(normalized_lines), False

        return "\n".join(normalized_lines), True

    @classmethod
    def generate(
        cls,
        prompts: list[GenerateDatasetSamplesPrompt],
        test_size: float = 0.2,
        batch_size: int = 4,
        sleep_seconds: float = 2.0,
        log_every_batches: int = 10,
        max_concurrency: int = 4,
        enable_dsl_validation: bool = True,
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
                raw_outputs = None
                for attempt in range(5):
                    try:
                        raw_outputs = chain.batch(
                            batch,
                            stop=None,
                            config={"max_concurrency": max(max_concurrency, 1)},
                        )
                        break
                    except Exception as rate_err:
                        if "429" in str(rate_err) or "rate_limit" in str(rate_err).lower():
                            wait = 2 ** attempt * 10
                            logger.warning(
                                f"Rate limit hit on batch {batch_idx + 1}, retrying in {wait}s "
                                f"(attempt {attempt + 1}/5)"
                            )
                            time.sleep(wait)
                            if attempt == 4:
                                raise
                        else:
                            raise

                if raw_outputs is None:
                    raise RuntimeError(f"No outputs returned for batch {batch_idx + 1}")

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
                        # Support question-generation prompt output format:
                        # {"caption_vn": "..."}
                        if (
                            "caption_vn" in sample_dict
                            and "instruction" not in sample_dict
                            and "answer" not in sample_dict
                        ):
                            sample_dict["instruction"] = str(prompt.document.caption_vn)
                            sample_dict["answer"] = str(sample_dict["caption_vn"])

                        sample_dict["image_dir"] = prompt.document.image_dir

                        if isinstance(sample_dict.get("answer"), list):
                            sample_dict["answer"] = "\n".join(sample_dict["answer"])

                        if enable_dsl_validation:
                            normalized_answer, is_valid_dsl = cls._normalize_and_validate_dsl(
                                sample_dict.get("answer", "")
                            )
                            if not is_valid_dsl:
                                logger.warning(
                                    "Skipping sample due to invalid DSL syntax after normalization. "
                                    f"instruction={sample_dict.get('instruction', '')[:120]}"
                                )
                                logger.debug(f"Invalid DSL: {sample_dict.get('answer', '')}")
                                continue
                            sample_dict["answer"] = normalized_answer

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


class InstructiveQuestionDatasetGenerator(DatasetGeneration):
    prompt_template_str = QUESTION_GENERATION_PROMPT

    @classmethod
    def post_process_datasets(cls, dataset: InstructDataset, test_size: float) -> InstructTrainTestSplit:
        return generation_utils.create_instruct_train_test_split([dataset], test_size=test_size, random_state=42)