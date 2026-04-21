from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from langchain_core.runnables import RunnableConfig
from loguru import logger


ProgressCallback = Callable[[dict[str, Any]], None]


class WorkflowProgressReporter:
    def __init__(self, callback: ProgressCallback | None = None, request_id: str | None = None):
        self._callback = callback
        self._request_id = request_id

    @staticmethod
    def _utcnow_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @classmethod
    def from_config(cls, config: RunnableConfig | None) -> WorkflowProgressReporter:
        if not isinstance(config, dict):
            return cls()

        configurable = config.get("configurable")
        if not isinstance(configurable, dict):
            return cls()

        callback = configurable.get("progress_callback")
        resolved_callback = callback if callable(callback) else None

        request_id = configurable.get("request_id")
        resolved_request_id = request_id if isinstance(request_id, str) and request_id else None

        return cls(callback=resolved_callback, request_id=resolved_request_id)

    def emit(self, event: str, **extra: Any) -> None:
        if self._callback is None:
            return

        payload: dict[str, Any] = {
            "event": event,
            "timestamp": self._utcnow_iso(),
        }
        if self._request_id is not None:
            payload["request_id"] = self._request_id
        payload.update(extra)

        try:
            self._callback(payload)
        except Exception:
            # Progress updates should never break orchestration execution.
            logger.exception("Could not emit orchestration progress event")

    def emit_step(self, step: str, status: str, **extra: Any) -> None:
        self.emit("orchestration.progress", step=step, status=status, **extra)

    def emit_stage(self, stage: str, status: str, **extra: Any) -> None:
        self.emit("orchestration.stage", stage=stage, status=status, **extra)
