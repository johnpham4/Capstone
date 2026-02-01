"""Problem Solver Agent - solves geometry problems using OpenAI."""

from typing import Dict, Any
from loguru import logger

from llm_src.domains.orchestration import Agent, AgentType, AgentState


class ProblemSolverAgent(Agent):
    """
    Solve geometry problems using OpenAI GPT-4/O1.

    Takes:
    - User question
    - DSL code (as context)
    - Conversation history

    Returns:
    - Step-by-step solution
    - Mathematical reasoning
    """

    SYSTEM_PROMPT = """Bạn là chuyên gia giải toán hình học.

Nhiệm vụ: Giải quyết các bài toán hình học Việt Nam với lời giải chi tiết, logic và dễ hiểu.

Định dạng lời giải:
1. Phân tích đề bài
2. Vẽ hình (nếu cần mô tả thêm)
3. Giải từng bước với giải thích rõ ràng
4. Kết luận

Bạn có thể nhận được:
- Mô tả hình học bằng DSL (S-expression)
- Lịch sử hội thoại

Hãy trả lời bằng tiếng Việt, rõ ràng và logic."""

    def __init__(self, model: str = "gpt-4o", temperature: float = 0.7):
        super().__init__(AgentType.PROBLEM_SOLVER)
        self.model = model
        self.temperature = temperature
        # OpenAI client will be injected via infrastructure
        self.openai_client = None

    def set_openai_client(self, client):
        """Inject OpenAI client (dependency injection)."""
        self.openai_client = client

    async def execute(self, state: AgentState) -> AgentState:
        """Solve geometry problem."""
        state.add_execution_step(self.name)

        if not self.openai_client:
            error_msg = "OpenAI client not configured"
            state.add_error(error_msg)
            state.solution_error = error_msg
            logger.error(error_msg)
            return state

        try:
            # Build context
            context_parts = []

            if state.dsl:
                context_parts.append(
                    f"Hình học được mô tả bằng DSL:\n```\n{state.dsl}\n```"
                )

            # Add user question
            user_question = state.user_input
            if state.history:
                # Get latest user message if available
                last_user = state.get_last_user_message()
                if last_user:
                    user_question = last_user

            # Build messages for OpenAI
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT}
            ]

            # Add context if available
            if context_parts:
                context_msg = "\n\n".join(context_parts)
                messages.append({
                    "role": "user",
                    "content": f"Thông tin bài toán:\n{context_msg}"
                })

            # Add user question
            messages.append({
                "role": "user",
                "content": f"Câu hỏi: {user_question}"
            })

            logger.info(f"Calling OpenAI {self.model} for problem solving")

            # Call OpenAI
            response = await self.openai_client.chat_completion(
                model=self.model,
                messages=messages,
                temperature=self.temperature
            )

            solution = response.get("content", "")

            if not solution.strip():
                raise ValueError("OpenAI returned empty solution")

            state.solution = solution

            # Extract steps if possible (simple heuristic)
            steps = self._extract_steps(solution)
            state.solution_steps = steps

            logger.info(f"Solution generated ({len(solution)} chars)")

        except Exception as e:
            error_msg = f"Problem solving failed: {str(e)}"
            state.add_error(error_msg)
            state.solution_error = error_msg
            logger.error(error_msg, exc_info=True)

        return state

    def _extract_steps(self, solution: str) -> list:
        """Extract solution steps (simple parsing)."""
        steps = []
        lines = solution.split('\n')

        for line in lines:
            line = line.strip()
            # Look for numbered steps
            if line and (
                line[0].isdigit() or
                line.startswith('Bước') or
                line.startswith('Step')
            ):
                steps.append(line)

        return steps if steps else [solution]

    def get_config(self) -> Dict[str, Any]:
        """Get agent configuration."""
        return {
            "type": self.agent_type.value,
            "model": self.model,
            "temperature": self.temperature
        }
