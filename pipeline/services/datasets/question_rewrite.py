import time

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from loguru import logger

from pipeline.domain.prompt_question import prompt as QUESTION_GENERATION_PROMPT
from src.config.settings.base import settings


class QuestionRewriteService:
    """Dedicated service: rewrite one Vietnamese geometry statement into one problem_vn."""

    system_prompt = (
        "You rewrite Vietnamese geometry givens into a natural exam-style problem. "
        "Return only valid JSON."
    )

    @staticmethod
    def _build_user_prompt(source_text: str) -> str:
        return QUESTION_GENERATION_PROMPT.replace("{{ extract }}", source_text)

    @staticmethod
    def _extract_problem_text(output: object) -> str:
        if isinstance(output, list) and output:
            output = output[0]

        if not isinstance(output, dict):
            return ""

        value = output.get("problem_vn", output.get("caption_vn", ""))
        return str(value).strip()

    @classmethod
    def rewrite_many(
        cls,
        source_texts: list[str],
        batch_size: int = 4,
        sleep_seconds: float = 2.0,
        log_every_batches: int = 10,
        max_concurrency: int = 4,
    ) -> list[str]:
        assert settings.OPENAI_API_KEY is not None, "OPENAI_API_KEY is required"

        llm = ChatOpenAI(
            model=settings.OPENAI_MODEL_ID,
            api_key=settings.OPENAI_API_KEY,
            max_tokens=300,
            temperature=0.4,
        )
        parser = JsonOutputParser()
        chain = llm | parser

        results: list[str] = [""] * len(source_texts)
        total_batches = (len(source_texts) + batch_size - 1) // batch_size if source_texts else 0

        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(source_texts))
            chunk = source_texts[start:end]

            if batch_idx > 0 and sleep_seconds > 0:
                time.sleep(sleep_seconds)

            messages_batch = [
                [
                    SystemMessage(content=cls.system_prompt),
                    HumanMessage(content=cls._build_user_prompt(text)),
                ]
                for text in chunk
            ]

            try:
                outputs = None
                for attempt in range(5):
                    try:
                        outputs = chain.batch(
                            messages_batch,
                            stop=None,
                            config={"max_concurrency": max(max_concurrency, 1)},
                        )
                        break
                    except Exception as err:
                        if "429" in str(err) or "rate_limit" in str(err).lower():
                            wait_seconds = (2**attempt) * 10
                            logger.warning(
                                "Rate limited on rewrite batch {}, retry in {}s ({}/5)",
                                batch_idx + 1,
                                wait_seconds,
                                attempt + 1,
                            )
                            time.sleep(wait_seconds)
                            continue
                        raise

                if outputs is None:
                    raise RuntimeError(f"No output for batch {batch_idx + 1}")

                for offset, output in enumerate(outputs):
                    results[start + offset] = cls._extract_problem_text(output)

                if (batch_idx + 1) % max(log_every_batches, 1) == 0 or batch_idx == 0:
                    logger.info(
                        "Question rewrite progress: {}/{} batches",
                        batch_idx + 1,
                        total_batches,
                    )

            except OutputParserException as err:
                logger.error(
                    "JSON parse error in rewrite batch {}: {}",
                    batch_idx + 1,
                    str(err),
                )
            except Exception as err:
                logger.error(
                    "Unexpected error in rewrite batch {}: {}: {}",
                    batch_idx + 1,
                    type(err).__name__,
                    str(err),
                )

        return results
