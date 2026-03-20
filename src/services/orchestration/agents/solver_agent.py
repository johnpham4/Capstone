from collections.abc import Generator

from openai import OpenAI
from loguru import logger

from src.config.settings.base import settings
from src.prompts import SOLVER_SYSTEM_PROMPT


class SolverAgent:
    def __init__(self):
        self.client = OpenAI(
            api_key=settings.OPENAI_API_KEY,
            timeout=60.0,
        )

    def execute(self, user_input: str) -> dict:
        try:
            solution = self._solve_problem(user_input)

            return {
                "solution": solution,
                "status": "success"
            }
        except Exception as e:
            logger.error(f"SolverAgent error: {e}")
            return {"error": str(e), "status": "failed"}

    def stream_solve(self, user_input: str) -> Generator[str, None, None]:
        """Yield solution tokens one chunk at a time (OpenAI stream=True)."""
        prompt = SOLVER_SYSTEM_PROMPT.format(problem=user_input)
        try:
            stream = self.client.chat.completions.create(
                model=settings.OPENAI_MODEL_ID,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=0.3,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as e:
            logger.error(f"SolverAgent stream error: {e}")
            yield f"\n[Error: {e}]"

    def _solve_problem(self, problem: str) -> str:
        prompt = SOLVER_SYSTEM_PROMPT.format(problem=problem)

        response = self.client.chat.completions.create(
            model=settings.OPENAI_MODEL_ID,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.3
        )

        content = response.choices[0].message.content
        return content
