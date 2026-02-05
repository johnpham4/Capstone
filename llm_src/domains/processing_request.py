from enum import StrEnum
from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import Field

from llm_src.domains.odm.nosql import NoSQLBaseDocument


class RequestStatus(StrEnum):
    """Status of processing request"""
    PENDING = "pending"
    PROCESSING_INPUT = "processing_input"
    PROCESSING_MODEL = "processing_model"
    GENERATING_DIAGRAM = "generating_diagram"
    COMPLETED = "completed"
    FAILED = "failed"


class ProcessingRequest(NoSQLBaseDocument):

    request_id: str
    status: RequestStatus = RequestStatus.PENDING
    user_input: str
    problem_text: Optional[str] = None

    # Processing results
    model_output: Optional[str] = None
    dsl_commands: Optional[list[str]] = None
    diagram_path: Optional[str] = None
    diagram_points: Optional[Dict[str, Any]] = None

    # Metadata
    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    class Settings:
        name = "processing_requests"

    # Business methods

    def update_status(self, status: RequestStatus, error_message: Optional[str] = None):
        """Update request status and save to database"""
        self.status = status
        self.updated_at = datetime.utcnow()
        if error_message:
            self.error_message = error_message
        if status == RequestStatus.COMPLETED or status == RequestStatus.FAILED:
            self.completed_at = datetime.utcnow()
        self.save()  # Auto save to database

    def update_model_output(self, model_output: str, dsl_commands: list[str]):
        """Update model output and save to database"""
        self.model_output = model_output
        self.dsl_commands = dsl_commands
        self.updated_at = datetime.utcnow()
        self.save()

    def update_diagram_result(self, diagram_path: str, diagram_points: dict):
        """Update diagram result and save to database"""
        self.diagram_path = diagram_path
        self.diagram_points = diagram_points
        self.updated_at = datetime.utcnow()
        self.save()

    # Class methods for querying

    @classmethod
    def get_by_id(cls, request_id: str) -> Optional["ProcessingRequest"]:
        """Get processing request by ID"""
        return cls.find(request_id=request_id)

    @classmethod
    def get_by_status(cls, status: RequestStatus, limit: int = 100) -> list["ProcessingRequest"]:
        """Get all requests with specific status"""
        # Note: bulk_find doesn't support limit, so we slice after
        all_requests = cls.bulk_find(status=status) or []
        return all_requests[:limit]

    @classmethod
    def delete_old_requests(cls, days: int = 7) -> int:
        """Delete old completed/failed requests"""
        from datetime import timedelta
        from llm_src.infrastructures.db.mongo import connection
        from llm_src.settings import settings

        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            collection = connection[settings.DATABASE_NAME][cls.Settings.name]
            result = collection.delete_many({
                "status": {"$in": [RequestStatus.COMPLETED.value, RequestStatus.FAILED.value]},
                "completed_at": {"$lt": cutoff_date}
            })
            return result.deleted_count
        except Exception as e:
            from loguru import logger
            logger.exception(f"Failed to delete old requests: {e}")
            raise
