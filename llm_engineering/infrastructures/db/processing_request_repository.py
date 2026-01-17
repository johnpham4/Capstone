from typing import Optional, List
from datetime import datetime, timedelta
from pymongo.collection import Collection
from loguru import logger

from llm_engineering.infrastructures.db.mongo import connection
from llm_engineering.domains.processing_request import ProcessingRequest, RequestStatus
from llm_engineering.settings import settings


class ProcessingRequestRepository:
    """Repository for managing processing requests in MongoDB"""

    def __init__(self):
        self.db = connection[settings.DATABASE_NAME]
        self.collection: Collection = self.db["processing_requests"]
        self._ensure_indexes()

    def _ensure_indexes(self):
        """Create indexes for efficient querying"""
        self.collection.create_index("request_id", unique=True)
        self.collection.create_index("status")
        self.collection.create_index("created_at")

    def create(self, request: ProcessingRequest) -> ProcessingRequest:
        """Create a new processing request"""
        try:
            request_dict = request.model_dump()
            self.collection.insert_one(request_dict)
            logger.info(f"Created processing request: {request.request_id}")
            return request
        except Exception as e:
            logger.exception(f"Failed to create processing request: {e}")
            raise

    def get_by_id(self, request_id: str) -> Optional[ProcessingRequest]:
        """Get a processing request by ID"""
        try:
            doc = self.collection.find_one({"request_id": request_id})
            if doc:
                doc.pop('_id', None)  # Remove MongoDB _id field
                return ProcessingRequest(**doc)
            return None
        except Exception as e:
            logger.exception(f"Failed to get processing request {request_id}: {e}")
            raise

    def update_status(
        self,
        request_id: str,
        status: RequestStatus,
        error_message: Optional[str] = None
    ) -> bool:
        """Update the status of a processing request"""
        try:
            update_data = {
                "status": status,
                "updated_at": datetime.utcnow()
            }

            if error_message:
                update_data["error_message"] = error_message

            if status in [RequestStatus.COMPLETED, RequestStatus.FAILED]:
                update_data["completed_at"] = datetime.utcnow()

            result = self.collection.update_one(
                {"request_id": request_id},
                {"$set": update_data}
            )

            logger.info(f"Updated status for request {request_id} to {status}")
            return result.modified_count > 0
        except Exception as e:
            logger.exception(f"Failed to update status for request {request_id}: {e}")
            raise

    def update_model_output(
        self,
        request_id: str,
        model_output: str,
        dsl_commands: List[str]
    ) -> bool:
        """Update model processing output"""
        try:
            result = self.collection.update_one(
                {"request_id": request_id},
                {"$set": {
                    "model_output": model_output,
                    "dsl_commands": dsl_commands,
                    "updated_at": datetime.utcnow()
                }}
            )
            logger.info(f"Updated model output for request {request_id}")
            return result.modified_count > 0
        except Exception as e:
            logger.exception(f"Failed to update model output for request {request_id}: {e}")
            raise

    def update_diagram_result(
        self,
        request_id: str,
        diagram_path: str,
        diagram_points: dict
    ) -> bool:
        """Update diagram generation result"""
        try:
            result = self.collection.update_one(
                {"request_id": request_id},
                {"$set": {
                    "diagram_path": diagram_path,
                    "diagram_points": diagram_points,
                    "updated_at": datetime.utcnow()
                }}
            )
            logger.info(f"Updated diagram result for request {request_id}")
            return result.modified_count > 0
        except Exception as e:
            logger.exception(f"Failed to update diagram result for request {request_id}: {e}")
            raise

    def get_by_status(self, status: RequestStatus, limit: int = 100) -> List[ProcessingRequest]:
        try:
            cursor = self.collection.find({"status": status}).limit(limit)
            requests = []
            for doc in cursor:
                doc.pop('_id', None)
                requests.append(ProcessingRequest(**doc))
            return requests
        except Exception as e:
            logger.exception(f"Failed to get requests by status {status}: {e}")
            raise

    def delete_old_requests(self, days: int = 7) -> int:
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            result = self.collection.delete_many({
                "status": {"$in": [RequestStatus.COMPLETED, RequestStatus.FAILED]},
                "completed_at": {"$lt": cutoff_date}
            })
            logger.info(f"Deleted {result.deleted_count} old requests")
            return result.deleted_count
        except Exception as e:
            logger.exception(f"Failed to delete old requests: {e}")
            raise


# Singleton instance
processing_request_repository = ProcessingRequestRepository()
