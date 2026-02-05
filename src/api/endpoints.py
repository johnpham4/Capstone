from fastapi import FastAPI

from src.api.routes.diagram import router as diagram_router
from src.api.routes.tasks import router as tasks_router
from src.api.routes.websocket import router as websocket_router


def register_routes(app: FastAPI) -> None:
    app.include_router(diagram_router, tags=["diagram"])
    app.include_router(tasks_router, tags=["tasks"])
    app.include_router(websocket_router, tags=["websocket"])
