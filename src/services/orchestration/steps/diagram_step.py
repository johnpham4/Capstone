import json
from loguru import logger
from src.infrastructures.celery.tasks import render_diagram_task
from src.config.settings import settings
from src.services.mock_responses import MOCK_DSL
from src.services.generators import VLLMOpenAIGenerator


class DiagramStep:

    def execute(
        self,
        user_input: str,
        dsl_prompt: str,
        llm_mock: bool = False,
        clean_problem: bool = True,
    ) -> dict:
        if llm_mock:
            dsl = MOCK_DSL
        else:
            generator = VLLMOpenAIGenerator()
            dsl = generator.generate_dsl(user_input, dsl_prompt, clean_problem=clean_problem)
            if not dsl or not dsl.strip():
                return {
                    "status": "failed",
                    "error_code": "DSL_GENERATION_ERROR",
                    "error": "DSL generator failed to generate DSL",
                }

        try:
            celery_task = render_diagram_task.apply_async(
                kwargs={
                    "task_id": f"orchestrator_{id(self)}",
                    "dsl": dsl,
                    "epochs": settings.DIAGRAM_OPTIMIZER_EPOCHS,
                    "dpi": 150,
                },
                queue=settings.DIAGRAM_QUEUE_NAME,
            )

            task_result = celery_task.get(
                timeout=settings.DIAGRAM_TASK_TIMEOUT_SECONDS,
                propagate=False,
            )

            if isinstance(celery_task.result, Exception):
                error_msg = str(celery_task.result)
                try:
                    parsed = json.loads(error_msg)
                    return {
                        "status": "failed",
                        "error_code": parsed.get("error_code", "DIAGRAM_TASK_ERROR"),
                        "error": parsed.get("message", error_msg),
                    }
                except (json.JSONDecodeError, TypeError):
                    return {"status": "failed", "error": error_msg}

            if isinstance(task_result, dict) and task_result.get("result"):
                result = task_result["result"]
                result["dsl"] = dsl
                return result

            if isinstance(task_result, dict):
                task_result["dsl"] = dsl
                return task_result

            return {
                "status": "failed",
                "error_code": "DIAGRAM_TASK_ERROR",
                "error": f"Unexpected worker result: {task_result}",
            }

        except Exception as e:
            logger.error(f"DiagramStep error: {e}")
            return {"status": "failed", "error_code": "DIAGRAM_TASK_ERROR", "error": str(e)}

