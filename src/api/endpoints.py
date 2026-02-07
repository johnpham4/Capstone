from fastapi import FastAPI

from src.api.routes.diagram import router as diagram_router


def register_routes(app: FastAPI) -> None:
    app.include_router(diagram_router, tags=["diagram"])
