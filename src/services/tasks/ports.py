from abc import ABC, abstractmethod
from typing import Any


class TaskQueuePort(ABC):
    @abstractmethod
    def queue_diagram_render(self, dsl: str, epochs: int, dpi: int) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_task_status(self, celery_task_id: str) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_workers_status(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_active_tasks(self) -> dict[str, Any]:
        raise NotImplementedError
