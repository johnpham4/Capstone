from __future__ import annotations

from typing import Any

from typing_extensions import TypedDict

from src.models.dto.orchestration import Mode


class WorkflowState(TypedDict):
    user_input: str
    image_base64: str | None
    ocr_text: str
    mode: Mode
    resolved_mode: Mode
    problem_statement: str
    diagram: dict[str, Any]
    solution: dict[str, Any]
    llm_mock: bool
    diagram_retry_count: int
