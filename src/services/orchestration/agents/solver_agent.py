from openai import OpenAI
from loguru import logger

from src.config.settings.base import settings


SOLVER_PROMPT = """Bạn là một giáo viên toán hình học.
Hãy giải bài toán sau một cách chi tiết, từng bước:

{problem}

Trả lời theo format:
- Đáp án cuối: [kết quả]
- Các bước giải:
  1. ...
  2. ...
"""


class SolverAgent:
    """Agent for math problem solving using OpenAI"""

    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def execute(self, user_input: str) -> dict:
        """Solve geometry problem"""
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
        """Call OpenAI to solve problem"""
        prompt = SOLVER_PROMPT.format(problem=problem)

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.3
        )

        return response.choices[0].message.content
