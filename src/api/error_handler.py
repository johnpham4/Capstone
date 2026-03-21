import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger

from src.api.exceptions import AppError


def register_error_handlers(app: FastAPI) -> None:
    """Register all global exception handlers on the app."""

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        """Handle custom AppError hierarchy."""
        body: dict = {
            "error": {
                "code": exc.error_code,
                "message": exc.message,
            }
        }

        # Attach request_id for server errors to aid debugging
        if exc.status_code >= 500:
            request_id = str(uuid.uuid4())
            body["error"]["request_id"] = request_id
            logger.error(
                f"[{request_id}] {exc.error_code}: {exc.message} "
                f"| path={request.url.path}"
            )
        else:
            logger.warning(f"{exc.error_code}: {exc.message} | path={request.url.path}")

        return JSONResponse(status_code=exc.status_code, content=body)

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Handle Pydantic request validation errors (422)."""
        errors = []
        for err in exc.errors():
            errors.append(
                {
                    "field": " → ".join(str(loc) for loc in err["loc"]),
                    "message": err["msg"],
                    "type": err["type"],
                }
            )

        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Request validation failed",
                    "details": errors,
                }
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Catch-all for unexpected errors — never leak stack traces."""
        request_id = str(uuid.uuid4())
        logger.exception(
            f"[{request_id}] Unhandled exception on {request.method} {request.url.path}"
        )

        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An unexpected error occurred",
                    "request_id": request_id,
                }
            },
        )
