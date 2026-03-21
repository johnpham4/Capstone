from abc import ABC, abstractmethod
import re
from langchain_openai import ChatOpenAI
from loguru import logger
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.exceptions import OutputParserException


from src.services.utils import misc
from src.services.diagram.dsl_parser import DSLParser
from src.config.settings.base import settings
from src.models.domain.training import Document
from src.models.domain.training import GenerateDatasetSamplesPrompt, Prompt
from src.models.domain.training import InstructDataset, InstructDatasetSample, InstructTrainTestSplit

from . import utils as generation_utils
from .prompt import prompt
import time

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
    def _normalize_and_validate_dsl(cls, answer: str) -> tuple[str, bool]:
        """Best-effort normalize DSL line parentheses and validate parseability."""
        if not isinstance(answer, str):
            return "", False

        raw_lines = [ln.strip() for ln in answer.splitlines() if ln.strip()]
        if not raw_lines:
            return "", False

        def _clean_prefix(line: str) -> str:
            # Remove common markdown/list prefixes from model outputs.
            line = line.strip().strip('"')
            if line.startswith("```"):
                return ""
            line = re.sub(r"^[-*•]\s+", "", line)
            line = re.sub(r"^\d+[\.)]\s+", "", line)
            line = re.sub(r"^→\s*", "", line)
            return line.strip()

        def _balance_line_parens(line: str) -> str:
            fixed = line
            # Ensure it is wrapped as one s-expression line.
            if not fixed.startswith("("):
                fixed = "(" + fixed
            if not fixed.endswith(")"):
                fixed = fixed + ")"

            # Prefer trimming extra closing parens at line end over rejecting.
            open_count = fixed.count("(")
            close_count = fixed.count(")")
            if close_count > open_count:
                extra = close_count - open_count
                while extra > 0 and fixed.endswith(")"):
                    fixed = fixed[:-1]
                    extra -= 1

            # Add missing closing parens when line is short by opens.
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
            fixed = _balance_line_parens(cleaned)
            normalized_lines.append(fixed)

        if not normalized_lines:
            return "", False

        # Global balance: trim extra trailing ')' first, then add missing ones.
        joined = "\n".join(normalized_lines)
        open_count = joined.count("(")
        close_count = joined.count(")")
        if close_count > open_count:
            extra = close_count - open_count
            while extra > 0 and joined.endswith(")"):
                joined = joined[:-1]
                extra -= 1
            # If still over-closed, it's likely malformed in the middle.
            if joined.count(")") > joined.count("("):
                return joined, False
        if open_count > close_count:
            joined = joined + (")" * (open_count - close_count))
            normalized_lines = [ln for ln in joined.splitlines() if ln.strip()]
        else:
            normalized_lines = [ln for ln in joined.splitlines() if ln.strip()]

        # Parse each line as an s-expression.
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
    def generate(cls, prompts: list[GenerateDatasetSamplesPrompt], test_size: float = 0.2, batch_size: int = 4) -> InstructTrainTestSplit:
        def _to_langchain(prompt: GenerateDatasetSamplesPrompt) -> list[BaseMessage]:
            return [
                SystemMessage(content=cls.get_system_prompt().content),
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

        messages_batch = [_to_langchain(p) for p in prompts]
        batches = list(misc.batch(messages_batch, size=batch_size))

        samples = []
        for batch_idx, batch in enumerate(batches):
            # Add delay between batches to avoid rate limit
            if batch_idx > 0:
                time.sleep(10)

            logger.info(f"Processing batch {batch_idx + 1}/{len(batches)}...")

            try:
                for attempt in range(5):
                    try:
                        raw_outputs = chain.batch(batch, stop=None, config={"max_concurrency": 1})
                        break
                    except Exception as rate_err:
                        if "429" in str(rate_err) or "rate_limit" in str(rate_err).lower():
                            wait = 2 ** attempt * 10
                            logger.warning(f"Rate limit hit on batch {batch_idx}, retrying in {wait}s (attempt {attempt+1}/5)")
                            time.sleep(wait)
                            if attempt == 4:
                                raise
                        else:
                            raise

                for idx, raw_output in enumerate(raw_outputs):
                    prompt_idx = batch_idx * batch_size + idx
                    if prompt_idx >= len(prompts):
                        continue

                    prompt = prompts[prompt_idx]

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
                        sample_dict['image_dir'] = prompt.document.image_dir

                        # Now convert to Pydantic model
                        if isinstance(sample_dict.get("answer"), list):
                            sample_dict["answer"] = "\n".join(sample_dict["answer"])

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
    prompt_template_str = prompt

    @classmethod
    def post_process_datasets(cls, dataset: InstructDataset, test_size: float) -> InstructTrainTestSplit:

        return generation_utils.create_instruct_train_test_split([dataset], test_size=test_size, random_state=42)
