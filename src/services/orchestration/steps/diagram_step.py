import json

import httpx
from loguru import logger

from src.infrastructures.celery.tasks import render_diagram_task
from src.config.settings.settings import settings
from src.services.mock_responses import MOCK_DSL


class DiagramStep:

    TIMEOUT = 120  # seconds to wait for worker result

    def execute(
        self,
        user_input: str,
        dsl_prompt: str,
        llm_mock: bool = False,
    ) -> dict:
        # ── 1. Resolve DSL ──────────────────────────────────
        if llm_mock:
            dsl = MOCK_DSL
        elif settings.LLM_ENDPOINT_URL:
            dsl = self._generate_dsl(user_input, dsl_prompt)
            if dsl is None:
                return {"status": "failed", "error": "LLM endpoint failed to generate DSL"}
        else:
            # Fallback: treat user_input as raw DSL
            dsl = user_input or ""

        if not dsl.strip():
            return {"status": "failed", "error": "DSL input is empty"}

        # ── 2. Render diagram via Celery ────────────────────
        try:
            celery_task = render_diagram_task.apply_async(
                kwargs={
                    "task_id": f"orchestrator_{id(self)}",
                    "dsl": dsl,
                    "epochs": 500,
                    "n_tries": 1,
                    "dpi": 150,
                },
                queue=settings.DIAGRAM_QUEUE_NAME,
            )

            task_result = celery_task.get(timeout=self.TIMEOUT, propagate=False)

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

            return {"status": "failed", "error": f"Unexpected worker result: {task_result}"}

        except Exception as e:
            logger.error(f"DiagramStep error: {e}")
            return {"status": "failed", "error": str(e)}

    # ── LLM endpoint call ───────────────────────────────────

    def _generate_dsl(self, user_input: str, dsl_prompt: str) -> str | None:
        """Call the LLM endpoint (ngrok/SageMaker) to convert problem text → DSL."""
        prompt = dsl_prompt.format(query=user_input)
        try:
            response = httpx.post(
                f"{settings.LLM_ENDPOINT_URL}/generate",
                json={"prompt": prompt},
                timeout=settings.LLM_ENDPOINT_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
            dsl = data.get("response", "").strip()
            if dsl:
                logger.info(f"LLM generated DSL ({len(dsl)} chars)")
            return dsl or None
        except Exception as e:
            logger.error(f"LLM endpoint call failed: {e}")
            return None
