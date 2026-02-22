from openai import OpenAI
from loguru import logger

from src.config.settings.base import settings
from src.prompts import SOLVER_SYSTEM_PROMPT


class SolverAgent:
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

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

    def _solve_problem(self, problem: str) -> str:
        prompt = SOLVER_SYSTEM_PROMPT.format(problem=problem)

        # ── OpenAI call (no prompt cache — hit rate ≈ 0 %) ───
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.3
        )

        content = response.choices[0].message.content
        return content
