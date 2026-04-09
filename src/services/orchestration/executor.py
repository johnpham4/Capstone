import time
from typing import Any

from loguru import logger

from src.models.dto.orchestration import Mode
from src.services.history import HistoryService
from src.services.orchestration.orchestrator import Orchestrator


class OrchestrationError(Exception):
    def __init__(self, code: str, message: str, request_id: str | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.request_id = request_id


class OrchestrationExecutor:
    def __init__(self, orchestrator: Orchestrator, history: HistoryService):
        self.orchestrator = orchestrator
        self.history = history

    @staticmethod
    def _validate_outputs(mode: Mode, result: dict[str, Any]) -> tuple[str, str] | None:
        if mode in {"diagram", "both"}:
            diagram = result.get("diagram")
            if not isinstance(diagram, dict):
                return ("DIAGRAM_GENERATION_ERROR", "Diagram result missing")
            if diagram.get("status") != "success":
                return ("DIAGRAM_GENERATION_ERROR", diagram.get("error", "Diagram generation failed"))

        if mode in {"solve", "both"}:
            solution = result.get("solution")
            if not isinstance(solution, dict):
                return ("SOLUTION_GENERATION_ERROR", "Solution result missing")
            if solution.get("status") != "success":
                return ("SOLUTION_GENERATION_ERROR", solution.get("error", "Solution generation failed"))

        return None

    async def _persist_artifacts(self, request_id: str, mode: Mode, result: dict[str, Any]) -> None:
        if result.get("diagram"):
            diagram = result["diagram"]
            await self.history.save_diagram(
                request_id=request_id,
                dsl=diagram.get("dsl", ""),
                image_url=diagram.get("s3_url") or diagram.get("image_url"),
                generation_time_ms=diagram.get("generation_time_ms"),
                render_time_ms=diagram.get("render_time_ms"),
            )

        if result.get("solution"):
            solution = result["solution"]
            await self.history.save_solution(
                request_id=request_id,
                content=solution.get("solution") or solution.get("content", ""),
            )

        resolved_mode = result.get("mode", mode)
        if resolved_mode != mode:
            await self.history.update_request_mode(request_id=request_id, mode=resolved_mode)

    async def execute(self, user_id: str, user_input: str, mode: Mode, llm_mock: bool = False) -> dict[str, Any]:
        request = await self.history.create_request(
            user_id=user_id,
            input_text=user_input,
            mode=mode,
        )
        request_id = request.id
        start = time.perf_counter()

        try:
            result = await self.orchestrator.execute(
                user_input=user_input,
                mode=mode,
                llm_mock=llm_mock,
            )
            resolved_mode = result.get("mode", mode)

            invalid = self._validate_outputs(resolved_mode, result)
            if invalid is not None:
                code, message = invalid
                raise OrchestrationError(code=code, message=message, request_id=request_id)

            await self._persist_artifacts(request_id=request_id, mode=mode, result=result)

            latency_ms = int((time.perf_counter() - start) * 1000)
            await self.history.complete_request(request_id, latency_ms=latency_ms)

            return {
                "status": "success",
                "request_id": request_id,
                "mode": resolved_mode,
                "diagram": result.get("diagram"),
                "solution": result.get("solution"),
            }

        except OrchestrationError:
            latency_ms = int((time.perf_counter() - start) * 1000)
            await self._mark_failed(request_id=request_id, latency_ms=latency_ms)
            raise
        except Exception as exc:
            latency_ms = int((time.perf_counter() - start) * 1000)
            await self._mark_failed(request_id=request_id, latency_ms=latency_ms)
            raise OrchestrationError(
                code="ORCHESTRATION_EXECUTION_ERROR",
                message=str(exc),
                request_id=request_id,
            ) from exc

    async def _mark_failed(self, request_id: str, latency_ms: int) -> None:
        try:
            await self.history.fail_request(request_id=request_id, latency_ms=latency_ms)
        except Exception:
            logger.exception(f"Could not mark request {request_id} as failed")
