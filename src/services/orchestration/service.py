import asyncio
import time
from typing import Any, Literal

from langgraph.graph import StateGraph, END
from loguru import logger
from typing_extensions import TypedDict

from src.models.dto.orchestration import Mode
from src.services.history import HistoryService
from src.services.orchestration.steps import DiagramStep, OcrStep, SolverStep


class OrchestrationError(Exception):
    def __init__(self, code: str, message: str, request_id: str | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.request_id = request_id


class _WorkflowState(TypedDict):
    user_input: str
    image_base64: str | None
    ocr_text: str
    mode: Mode
    resolved_mode: Mode
    problem_statement: str
    diagram: dict
    solution: dict
    llm_mock: bool


class OrchestrationService:
    def __init__(self, diagram_prompt: str):
        self.diagram_prompt = diagram_prompt
        self._ocr_step: OcrStep | None = None
        self._diagram_step = DiagramStep()
        self._solver_step: SolverStep | None = None
        self._workflow = self._build_workflow()

    # ── LangGraph workflow ────────────────────────────────────

    def _build_workflow(self):
        g = StateGraph(_WorkflowState)
        g.add_node("ocr", self._ocr_node)
        g.add_node("parse", self._parse_node)
        g.add_node("diagram", self._diagram_node)
        g.add_node("solve", self._solve_node)

        g.set_entry_point("ocr")
        g.add_edge("ocr", "parse")
        g.add_conditional_edges(
            "parse",
            lambda s: s["resolved_mode"],
            {"diagram": "diagram", "solve": "solve", "both": "diagram"},
        )
        g.add_conditional_edges(
            "diagram",
            lambda s: "solve" if s["resolved_mode"] == "both" else END,
            {"solve": "solve", END: END},
        )
        g.add_edge("solve", END)
        return g.compile()

    def _ocr_node(self, state: _WorkflowState) -> _WorkflowState:
        image = state.get("image_base64")
        if not image:
            return state

        hint = state.get("user_input", "")
        try:
            if self._ocr_step is None:
                self._ocr_step = OcrStep()
            result = self._ocr_step.execute(image, hint=hint)
        except Exception as exc:
            logger.warning(f"OCR unavailable: {exc}")
            return state

        if result["status"] == "success" and result["extracted_text"]:
            extracted = result["extracted_text"]
            state["ocr_text"] = extracted
            existing = state.get("user_input", "").strip()
            if existing:
                state["user_input"] = f"{existing}\n\n[OCR từ ảnh]:\n{extracted}"
            else:
                state["user_input"] = extracted
            logger.info(f"OCR extracted text merged into user_input")
        else:
            logger.warning(f"OCR failed or empty: {result.get('error', 'unknown')}")

        return state

    @staticmethod
    def _parse_node(state: _WorkflowState) -> _WorkflowState:
        state["resolved_mode"] = state.get("mode", "diagram")
        state["problem_statement"] = state.get("user_input", "")
        return state

    def _diagram_node(self, state: _WorkflowState) -> _WorkflowState:
        state["diagram"] = self._diagram_step.execute(
            state["problem_statement"],
            self.diagram_prompt,
            llm_mock=state.get("llm_mock", False),
        )
        return state

    def _solve_node(self, state: _WorkflowState) -> _WorkflowState:
        solve_input = state["problem_statement"]
        if state.get("diagram") and state["diagram"].get("dsl"):
            solve_input += f"\n\n[Diagram DSL: {state['diagram']['dsl']}]"
        try:
            if self._solver_step is None:
                self._solver_step = SolverStep()
            state["solution"] = self._solver_step.execute(solve_input)
        except Exception as exc:
            logger.warning(f"Solver unavailable: {exc}")
            state["solution"] = {
                "status": "failed",
                "error": str(exc),
            }
        return state

    # ── Public API ────────────────────────────────────────────

    async def execute(
        self,
        user_id: str,
        user_input: str,
        mode: Mode,
        history: HistoryService,
        llm_mock: bool = False,
        image_base64: str | None = None,
    ) -> dict[str, Any]:
        # Upload source image to S3 if provided
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
        start = time.perf_counter()

        try:
            result = await self._run_workflow(user_input, mode, llm_mock, image_base64)
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

            return {
                "status": "success",
                "request_id": request_id,
                "mode": resolved_mode,
                "ocr_text": ocr_text,
                "diagram": result.get("diagram"),
                "solution": result.get("solution"),
            }

        except OrchestrationError:
            await self._mark_failed(history, request_id, time.perf_counter() - start)
            raise
        except Exception as exc:
            await self._mark_failed(history, request_id, time.perf_counter() - start)
            raise OrchestrationError(
                code="ORCHESTRATION_EXECUTION_ERROR",
                message=str(exc),
                request_id=request_id,
            ) from exc

    # ── Internal helpers ──────────────────────────────────────

    async def _run_workflow(self, user_input: str, mode: Mode, llm_mock: bool, image_base64: str | None = None) -> dict:
        initial: _WorkflowState = {
            "user_input": user_input,
            "image_base64": image_base64,
            "ocr_text": "",
            "mode": mode,
            "resolved_mode": mode,
            "problem_statement": user_input,
            "diagram": {},
            "solution": {},
            "llm_mock": llm_mock,
        }
        final = await asyncio.to_thread(self._workflow.invoke, initial)

        result: dict = {"mode": final.get("resolved_mode", mode)}
        if final.get("ocr_text"):
            result["ocr_text"] = final["ocr_text"]
        if final.get("diagram"):
            result["diagram"] = final["diagram"]
        if final.get("solution"):
            result["solution"] = final["solution"]
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
