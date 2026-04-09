from loguru import logger

from src.services.diagram.generation import DiagramService
from src.services.mock_responses import MOCK_DSL


class DiagramAgent:

    def __init__(self):
        self.diagram_service = DiagramService()

    def execute(
        self,
        user_input: str,
        dsl_prompt: str,
        llm_mock: bool = False,
    ) -> dict:
        _ = dsl_prompt
        dsl_to_render = MOCK_DSL if llm_mock else (user_input or "")

        if not dsl_to_render:
            return {
                "status": "failed",
                "error": "DSL input is empty",
            }

        try:
            return self.diagram_service.generate_and_render(
                task_id=f"orchestrator_{id(self)}",
                dsl=dsl_to_render,
                epochs=500,
                n_tries=1,
                dpi=150,
                timeout=60,
            )

        except Exception as e:
            logger.error(f"DiagramAgent error: {e}")
            return {"error": str(e), "status": "failed"}
