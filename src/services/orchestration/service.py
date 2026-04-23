import asyncio
import time
from typing import Any, Literal

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from loguru import logger

from src.models.dto.orchestration import Mode
from src.services.history import HistoryService
from src.services.orchestration.nodes import WorkflowNodes
from src.services.orchestration.progress import ProgressCallback, WorkflowProgressReporter
from src.services.orchestration.workflow_state import WorkflowState


class OrchestrationError(Exception):
    def __init__(self, code: str, message: str, request_id: str | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.request_id = request_id

class OrchestrationService:
    def __init__(self, diagram_prompt: str):
        self.diagram_prompt = diagram_prompt
        self._nodes = WorkflowNodes(diagram_prompt=diagram_prompt)
        self._workflow = self._build_workflow()

    def _build_workflow(self):
        g = StateGraph(WorkflowState)
        g.add_node("ocr", self._nodes.ocr_node)
        g.add_node("parse", self._nodes.parse_node)
        g.add_node("diagram", self._nodes.diagram_node)
        g.add_node("diagram_retry", self._nodes.diagram_retry_node)
        g.add_node("solve", self._nodes.solve_node)

        g.set_entry_point("ocr")
        g.add_edge("ocr", "parse")
        g.add_conditional_edges(
            "parse",
            self._route_after_parse,
            {"diagram": "diagram", "solve": "solve", "both": "diagram"},
        )
        g.add_conditional_edges(
            "diagram",
            self._route_after_diagram,
            {"retry": "diagram_retry", "solve": "solve", "end": END},
        )
        g.add_conditional_edges(
            "diagram_retry",
            self._route_after_diagram_retry,
            {"solve": "solve", "end": END},
        )
        g.add_edge("solve", END)
        return g.compile()

    @staticmethod
    def _route_after_parse(state: WorkflowState) -> Mode:
        return state["resolved_mode"]

    @staticmethod
    def _should_retry_diagram(state: WorkflowState) -> bool:
        diagram = state.get("diagram") or {}
        if diagram.get("status") == "success":
            return False

        # A single retry is enough for transient DSL generation failures.
        if state.get("diagram_retry_count", 0) >= 1:
            return False

        if state.get("llm_mock", False):
            return False

        retryable_codes = {"DSL_GENERATION_ERROR", "DSL_INPUT_REQUIRED", "DSL_EMPTY"}
        error_code = diagram.get("error_code")
        return error_code in retryable_codes or error_code is None

    def _route_after_diagram(self, state: WorkflowState) -> Literal["retry", "solve", "end"]:
        if self._should_retry_diagram(state):
            return "retry"
        return "solve" if state["resolved_mode"] == "both" else "end"

    @staticmethod
    def _route_after_diagram_retry(state: WorkflowState) -> Literal["solve", "end"]:
        return "solve" if state["resolved_mode"] == "both" else "end"

    # ── Public API ────────────────────────────────────────────

    async def execute(
        self,
        user_id: str,
        user_input: str,
        mode: Mode,
        history: HistoryService,
        llm_mock: bool = False,
        image_base64: str | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        # Upload source image to configured storage backend if provided.
        source_image_url = None
        if image_base64:
            source_image_url = history.upload_source_image(image_base64)

        request = await history.create_request(
            user_id=user_id,
            input_text=user_input or "(image upload)",
            mode=mode,
            source_image_url=source_image_url,
        )
        request_id = request.id
        # Log incoming orchestration request for debugging/tracing
        try:
            logger.info(f"Orchestration request {request_id} received: mode={mode}, user_input={user_input}")
        except Exception:
            # Ensure logging errors don't break orchestration
            logger.exception("Failed to log incoming orchestration request")
        start = time.perf_counter()
        reporter = WorkflowProgressReporter(callback=progress_callback, request_id=request_id)

        reporter.emit("orchestration.started", mode=mode)

        try:
            result = await self._run_workflow(
                user_input,
                mode,
                llm_mock,
                image_base64,
                request_id=request_id,
                progress_callback=progress_callback,
            )
            resolved_mode = result.get("mode", mode)

            error = self._validate_outputs(resolved_mode, result)
            if error is not None:
                code, message = error
                raise OrchestrationError(code=code, message=message, request_id=request_id)

            # Persist OCR text if extracted
            ocr_text = result.get("ocr_text")
            if ocr_text:
                await history.update_ocr_text(request_id, ocr_text)

            await self._persist_artifacts(history, request_id, mode, result)

            latency_ms = int((time.perf_counter() - start) * 1000)
            await history.complete_request(request_id, latency_ms=latency_ms)

            reporter.emit("orchestration.completed", mode=resolved_mode, latency_ms=latency_ms)

            return {
                "status": "success",
                "request_id": request_id,
                "mode": resolved_mode,
                "ocr_text": ocr_text,
                "diagram": result.get("diagram"),
                "solution": result.get("solution"),
            }

        except OrchestrationError as exc:
            await self._mark_failed(history, request_id, time.perf_counter() - start)
            reporter.emit("orchestration.failed", error_code=exc.code, message=exc.message)
            raise
        except Exception as exc:
            await self._mark_failed(history, request_id, time.perf_counter() - start)
            reporter.emit("orchestration.failed", error_code="ORCHESTRATION_EXECUTION_ERROR", message=str(exc))
            raise OrchestrationError(
                code="ORCHESTRATION_EXECUTION_ERROR",
                message=str(exc),
                request_id=request_id,
            ) from exc

    # ── Internal helpers ──────────────────────────────────────

    async def _run_workflow(
        self,
        user_input: str,
        mode: Mode,
        llm_mock: bool,
        image_base64: str | None = None,
        request_id: str | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> dict:
        initial = self._build_initial_state(user_input, image_base64, mode, llm_mock)
        workflow_config = self._build_workflow_config(request_id, progress_callback)
        final = await asyncio.to_thread(self._workflow.invoke, initial, workflow_config)

        # Log the raw input preserved for the solver and the cleaned DSL problem
        # produced by the parse node so they appear in the backend terminal for
        # easier debugging and tracing.
        try:
            user_before = final.get("problem_statement") or initial.get("problem_statement")
            dsl_problem = final.get("dsl_problem") or ""
            logger.info(
                f"Orchestration parse result request={request_id} user_input_before_parse={user_before!r} dsl_problem={dsl_problem!r}"
            )
        except Exception:
            logger.exception("Failed to log orchestration final parse values")

        return self._collect_workflow_result(final, mode)

    @staticmethod
    def _build_initial_state(
        user_input: str,
        image_base64: str | None,
        mode: Mode,
        llm_mock: bool,
    ) -> WorkflowState:
        return {
            "user_input": user_input,
            "image_base64": image_base64,
            "ocr_text": "",
            "mode": mode,
            "resolved_mode": mode,
            "problem_statement": user_input,
            "diagram": {},
            "solution": {},
            "llm_mock": llm_mock,
            "diagram_retry_count": 0,
        }

    @staticmethod
    def _build_workflow_config(
        request_id: str | None,
        progress_callback: ProgressCallback | None,
    ) -> RunnableConfig | None:
        # Always supply a runnable config so nodes can access request_id for logging.
        # The WorkflowProgressReporter will ignore a non-callable progress_callback.
        return {
            "configurable": {
                "request_id": request_id,
                "progress_callback": progress_callback,
            }
        }

    @staticmethod
    def _collect_workflow_result(final: dict[str, Any], mode: Mode) -> dict[str, Any]:
        result: dict[str, Any] = {"mode": final.get("resolved_mode", mode)}
        for key in ("ocr_text", "diagram", "solution"):
            value = final.get(key)
            if value:
                result[key] = value
        return result

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

    async def _persist_artifacts(self, history: HistoryService, request_id: str, mode: Mode, result: dict[str, Any]) -> None:
        if result.get("diagram"):
            d = result["diagram"]
            await history.save_diagram(
                request_id=request_id,
                dsl=d.get("dsl", ""),
                image_url=d.get("s3_url") or d.get("image_url"),
                generation_time_ms=d.get("generation_time_ms"),
                render_time_ms=d.get("render_time_ms"),
            )

        if result.get("solution"):
            s = result["solution"]
            await history.save_solution(
                request_id=request_id,
                content=s.get("solution") or s.get("content", ""),
            )

        resolved_mode = result.get("mode", mode)
        if resolved_mode != mode:
            await history.update_request_mode(request_id=request_id, mode=resolved_mode)

    async def _mark_failed(self, history: HistoryService, request_id: str, elapsed: float) -> None:
        try:
            latency_ms = int(elapsed * 1000)
            await history.fail_request(request_id=request_id, latency_ms=latency_ms)
        except Exception:
            logger.exception(f"Could not mark request {request_id} as failed")
