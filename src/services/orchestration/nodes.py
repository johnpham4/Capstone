from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from loguru import logger

from src.services.orchestration.progress import WorkflowProgressReporter
from src.services.orchestration.steps import DiagramStep, OcrStep, SolverStep
from src.services.orchestration.workflow_state import WorkflowState


class WorkflowNodes:
    def __init__(self, diagram_prompt: str):
        self.diagram_prompt = diagram_prompt
        self._ocr_step: OcrStep | None = None
        self._diagram_step = DiagramStep()
        self._solver_step: SolverStep | None = None

    @staticmethod
    def _emit_step_result(
        reporter: WorkflowProgressReporter,
        *,
        step: str,
        result: dict[str, Any],
        **extra: Any,
    ) -> bool:
        if result.get("status") == "success":
            reporter.emit_step(step, "succeeded", **extra)
            return True

        reporter.emit_step(
            step,
            "failed",
            error_code=result.get("error_code", f"{step.upper()}_ERROR"),
            message=result.get("error", f"{step} failed"),
            **extra,
        )
        return False

    @staticmethod
    def _merge_ocr_into_input(existing_text: str, ocr_text: str) -> str:
        existing = str(existing_text or "").strip()
        if existing:
            return f"{existing}\n\n[OCR từ ảnh]:\n{ocr_text}"
        return ocr_text

    def ocr_node(self, state: WorkflowState, config: RunnableConfig | None = None) -> WorkflowState:
        reporter = WorkflowProgressReporter.from_config(config)
        reporter.emit_step("ocr", "started")
        reporter.emit_stage("extracting_text", "started")

        image = state.get("image_base64")
        if not image:
            reporter.emit_step("ocr", "skipped", reason="no_image")
            reporter.emit_stage("extracting_text", "skipped", reason="no_image")
            return state

        hint = state.get("user_input", "")
        try:
            if self._ocr_step is None:
                self._ocr_step = OcrStep()
            result = self._ocr_step.execute(image, hint=hint)
        except Exception as exc:
            logger.warning(f"OCR unavailable: {exc}")
            reporter.emit_step("ocr", "failed", error_code="OCR_UNAVAILABLE", message=str(exc))
            reporter.emit_stage("extracting_text", "failed", error_code="OCR_UNAVAILABLE", message=str(exc))
            return state

        if result["status"] == "success" and result["extracted_text"]:
            extracted = result["extracted_text"]
            state["ocr_text"] = extracted
            state["user_input"] = self._merge_ocr_into_input(state.get("user_input", ""), extracted)
            logger.info("OCR extracted text merged into user_input")
            reporter.emit_step("ocr", "succeeded", extracted_chars=len(extracted))
            reporter.emit_stage("extracting_text", "completed", extracted_chars=len(extracted))
        else:
            logger.warning(f"OCR failed or empty: {result.get('error', 'unknown')}")
            reporter.emit_step(
                "ocr",
                "failed",
                error_code=result.get("error_code", "OCR_FAILED"),
                message=result.get("error", "unknown"),
            )
            reporter.emit_stage(
                "extracting_text",
                "failed",
                error_code=result.get("error_code", "OCR_FAILED"),
                message=result.get("error", "unknown"),
            )

        return state

    @staticmethod
    def parse_node(state: WorkflowState, config: RunnableConfig | None = None) -> WorkflowState:
        reporter = WorkflowProgressReporter.from_config(config)
        reporter.emit_step("parse", "started")
        reporter.emit_stage("analyzing_problem", "started")
        state["resolved_mode"] = state.get("mode", "diagram")
        state["problem_statement"] = state.get("user_input", "")
        reporter.emit_step("parse", "succeeded", mode=state["resolved_mode"])
        reporter.emit_stage("analyzing_problem", "completed", mode=state["resolved_mode"])
        return state

    def diagram_node(self, state: WorkflowState, config: RunnableConfig | None = None) -> WorkflowState:
        reporter = WorkflowProgressReporter.from_config(config)
        reporter.emit_step("diagram", "started")
        reporter.emit_stage("generating_diagram", "started", attempt=1)
        state["diagram"] = self._diagram_step.execute(
            state["problem_statement"],
            self.diagram_prompt,
            llm_mock=state.get("llm_mock", False),
            clean_problem=True,
        )
        diagram = state.get("diagram") or {}
        success = self._emit_step_result(reporter, step="diagram", result=diagram)
        if success:
            reporter.emit_stage("generating_diagram", "completed", attempt=1)
        else:
            reporter.emit_stage(
                "generating_diagram",
                "failed",
                attempt=1,
                error_code=diagram.get("error_code", "DIAGRAM_ERROR"),
                message=diagram.get("error", "diagram failed"),
            )
        return state

    def diagram_retry_node(self, state: WorkflowState, config: RunnableConfig | None = None) -> WorkflowState:
        reporter = WorkflowProgressReporter.from_config(config)
        logger.info("Retrying diagram generation with raw problem text")
        state["diagram_retry_count"] = state.get("diagram_retry_count", 0) + 1
        attempt = state["diagram_retry_count"] + 1
        reporter.emit_step("diagram_retry", "started", retry_count=state["diagram_retry_count"])
        reporter.emit_stage(
            "generating_diagram",
            "started",
            attempt=attempt,
            retry_count=state["diagram_retry_count"],
            retried=True,
        )
        state["diagram"] = self._diagram_step.execute(
            state["problem_statement"],
            self.diagram_prompt,
            llm_mock=state.get("llm_mock", False),
            clean_problem=False,
        )
        diagram = state.get("diagram") or {}
        success = self._emit_step_result(
            reporter,
            step="diagram_retry",
            result=diagram,
            retry_count=state["diagram_retry_count"],
        )
        if success:
            reporter.emit_stage(
                "generating_diagram",
                "completed",
                attempt=attempt,
                retry_count=state["diagram_retry_count"],
                retried=True,
            )
        else:
            reporter.emit_stage(
                "generating_diagram",
                "failed",
                attempt=attempt,
                retry_count=state["diagram_retry_count"],
                retried=True,
                error_code=diagram.get("error_code", "DIAGRAM_ERROR"),
                message=diagram.get("error", "diagram retry failed"),
            )
        return state

    def solve_node(self, state: WorkflowState, config: RunnableConfig | None = None) -> WorkflowState:
        reporter = WorkflowProgressReporter.from_config(config)
        reporter.emit_step("solve", "started")
        reporter.emit_stage("solving_problem", "started")

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
            reporter.emit_step("solve", "failed", error_code="SOLUTION_GENERATION_ERROR", message=str(exc))
            reporter.emit_stage("solving_problem", "failed", error_code="SOLUTION_GENERATION_ERROR", message=str(exc))
            return state

        solution = state.get("solution") or {}
        success = self._emit_step_result(reporter, step="solve", result=solution)
        if success:
            reporter.emit_stage("solving_problem", "completed")
        else:
            reporter.emit_stage(
                "solving_problem",
                "failed",
                error_code=solution.get("error_code", "SOLUTION_GENERATION_ERROR"),
                message=solution.get("error", "solution failed"),
            )
        return state
