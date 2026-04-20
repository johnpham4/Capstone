import httpx

from abc import ABC, abstractmethod
from loguru import logger
import httpx
from src.config.settings import settings
from src.services.utils.question_cleaning import clean_problem_section

class DSLGenerator(ABC):
    @abstractmethod
    def generate_dsl(self, user_input: str, dsl_prompt: str, clean_problem: bool = True) -> str | None:
        pass

class VLLMOpenAIGenerator(DSLGenerator):
    MAX_NEW_TOKENS = 256
    TEMPERATURE = 0.0
    TOP_P = 1.0
    REPETITION_PENALTY = 1.08
    TIMEOUT = 120


    def generate_dsl(self, user_input: str, dsl_prompt: str, clean_problem: bool = True) -> str | None:
        raw_input = str(user_input or "").strip()
        if clean_problem:
            cleaned, _ = clean_problem_section(raw_input)
            query_text = cleaned or raw_input
        else:
            query_text = raw_input

        prompt = dsl_prompt.format(query=query_text)
        payload = {
            "model": settings.VLLM_MODEL_ID,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.MAX_NEW_TOKENS,
            "temperature": self.TEMPERATURE,
            "top_p": self.TOP_P,
            "repetition_penalty": self.REPETITION_PENALTY,
        }

        try:
            response = httpx.post(
                self._vllm_chat_url(),
                json=payload,
                timeout=self.TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
            dsl = self._extract_chat_text(data)
            if dsl:
                logger.info(f"vLLM generated DSL ({len(dsl)} chars)")
            return dsl or None
        except Exception as e:
            logger.error(f"vLLM endpoint call failed: {e}")
            return None

    @staticmethod
    def _vllm_chat_url() -> str:
        base = settings.VLLM_BASE_URL.rstrip("/")
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/v1/chat/completions"

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
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
            return "".join(parts).strip()
        return str(content).strip()


