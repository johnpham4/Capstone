from loguru import logger

from src.services.diagram.generation import DiagramService


class DiagramAgent:

    def __init__(self):
        self.diagram_service = DiagramService()

    def execute(self, user_input: str, dsl_prompt: str, llm_mock: bool = False) -> dict:
        _ = (dsl_prompt, llm_mock)
        try:
            return self.diagram_service.generate_and_render(
                task_id=f"orchestrator_{id(self)}",
                dsl=user_input,
                epochs=500,
                n_tries=1,
                dpi=150,
                timeout=60,
            )

        except Exception as e:
            logger.error(f"DiagramAgent error: {e}")
            return {"error": str(e), "status": "failed"}
