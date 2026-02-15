from fastapi import FastAPI

from src.api.routes.diagram import router as diagram_router
from src.api.routes.auth import router as auth_router
from src.api.routes.orchestration import router as orchestration_router
from src.api.routes.tasks import router as tasks_router


def register_routes(app: FastAPI) -> None:
    app.include_router(auth_router, tags=["auth"])
    app.include_router(diagram_router, tags=["diagram"])
    app.include_router(orchestration_router, tags=["orchestration"])
    app.include_router(tasks_router, tags=["tasks"])
