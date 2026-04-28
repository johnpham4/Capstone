from abc import ABC, abstractmethod
from typing import Any, Tuple

import json
import httpx
import boto3
from loguru import logger

from src.config.settings import settings
from src.services.utils.question_cleaning import clean_problem_section


class DSLGenerator(ABC):
    @abstractmethod
    def generate_dsl(
        self,
        user_input: str,
        dsl_prompt: str,
        clean_problem: bool = True,
    ) -> str | None:
        pass


class BaseVLLMGenerator(DSLGenerator):
    MAX_NEW_TOKENS = 256
    TEMPERATURE = 0.0
    TOP_P = 1.0
    REPETITION_PENALTY = 1.08
    DEFAULT_MODEL = "text2diagram"

    def generate_dsl(self, user_input: str, dsl_prompt: str, clean_problem: bool = True) -> str | None:
        prompt = self._build_prompt(user_input, dsl_prompt, clean_problem)

        dsl, status = self._try_openai_chat(prompt)
        if dsl:
            return dsl

        if status != 422:
            return None

        return self._try_custom_prompt(prompt)

    @staticmethod
    def _build_prompt(user_input: str, dsl_prompt: str, clean_problem: bool) -> str:
        raw_input = str(user_input or "").strip()
        query_text = raw_input
        if clean_problem:
            cleaned, _ = clean_problem_section(raw_input)
            query_text = cleaned or raw_input
        return dsl_prompt.format(query=query_text)

    def _try_openai_chat(self, prompt: str) -> Tuple[str | None, int | None]:
        payload = {
            "model": settings.HF_MODEL_ID or self.DEFAULT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.MAX_NEW_TOKENS,
            "temperature": self.TEMPERATURE,
            "top_p": self.TOP_P,
            "repetition_penalty": self.REPETITION_PENALTY,
        }

        try:
            data = self._post_json(payload)
            dsl = self._extract_chat_text(data)
            if dsl:
                logger.info(f"vLLM generated DSL ({len(dsl)} chars)")
            return dsl or None, None

        except Exception as exc:
            status = getattr(exc, "response", None)
            status_code = status.status_code if status else None
            logger.error(f"OpenAI schema failed ({status_code}): {exc}")
            return None, status_code

    def _try_custom_prompt(self, prompt: str) -> str | None:
        payload = {
            "prompt": prompt,
            "max_new_tokens": self.MAX_NEW_TOKENS,
        }

        try:
            data = self._post_json(payload)
            dsl = str(data.get("response") or "").strip()
            return dsl or None

        except Exception as exc:
            logger.error(f"Custom schema failed: {exc}")
            return None

    @staticmethod
    def _extract_chat_text(data: dict) -> str:
        choices = data.get("choices") or []
        if not choices:
            return ""

        message = (choices[0] or {}).get("message") or {}
        content = message.get("content", "")

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            return "".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict)
            ).strip()

        return str(content).strip()

    @abstractmethod
    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        pass


# Local / HTTP (ngrok, vLLM server)
class VLLMOpenAIGenerator(BaseVLLMGenerator):
    TIMEOUT = 120

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = httpx.post(
            self._vllm_chat_url(),
            json=payload,
            timeout=self.TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _vllm_chat_url() -> str:
        base = settings.LLM_ENDPOINT_URL.rstrip("/")
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/v1/chat/completions"


# SageMaker (boto3)
class VLLMSagemakerGenerator(BaseVLLMGenerator):
    def __init__(self):
        self.client = boto3.client(
            "sagemaker-runtime",
            region_name=settings.AWS_REGION,
        )
        self.endpoint_name = settings.SAGEMAKER_ENDPOINT_NAME

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = self.client.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType="application/json",
            Body=json.dumps(payload),
        )

        body = response["Body"].read().decode("utf-8")
        data = json.loads(body)

        return data if isinstance(data, dict) else {}


class DSLGeneratorFactory:
    @staticmethod
    def create() -> DSLGenerator:
        provider = (settings.LLM_PROVIDER or "local").lower()

        if provider == "sagemaker":
            logger.info("Using SageMaker LLM")
            return VLLMSagemakerGenerator()

        if provider == "local":
            logger.info("Using local vLLM (HTTP)")
            return VLLMOpenAIGenerator()

        raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")
