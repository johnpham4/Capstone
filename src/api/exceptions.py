class AppError(Exception):
    """Base application error with status code + machine-readable code."""

    def __init__(
        self,
        message: str = "An unexpected error occurred",
        status_code: int = 500,
        error_code: str = "INTERNAL_ERROR",
    ) -> None:
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        super().__init__(message)


class ValidationError(AppError):
    """Input validation failed (400)."""

    def __init__(self, message: str = "Validation error") -> None:
        super().__init__(message, status_code=400, error_code="VALIDATION_ERROR")


class NotFoundError(AppError):
    """Resource does not exist (404)."""

    def __init__(self, resource: str = "Resource") -> None:
        super().__init__(
            message=f"{resource} not found",
            status_code=404,
            error_code="NOT_FOUND",
        )


class ForbiddenError(AppError):
    """User is authenticated but not authorized (403)."""

    def __init__(self, message: str = "Forbidden") -> None:
        super().__init__(message, status_code=403, error_code="FORBIDDEN")


class RateLimitError(AppError):
    """Too many requests (429)."""
    def __init__(self, message: str = "Rate limit exceeded") -> None:
        super().__init__(message, status_code=429, error_code="RATE_LIMITED")


class DSLGenerationError(AppError):
    """SageMaker / LLM DSL generation failed (502)."""

    def __init__(self, message: str = "DSL generation failed") -> None:
        super().__init__(message, status_code=502, error_code="DSL_GENERATION_ERROR")


class RenderingError(AppError):
    """Geometry renderer (PyTorch optimizer) failed (500)."""

    def __init__(self, message: str = "Diagram rendering failed") -> None:
        super().__init__(message, status_code=500, error_code="RENDERING_ERROR")


class ExternalServiceError(AppError):
    """Upstream dependency (OpenAI, SageMaker, Redis…) unavailable (502)."""

    def __init__(self, service: str = "External service") -> None:
        super().__init__(
            message=f"{service} is unavailable",
            status_code=502,
            error_code="EXTERNAL_SERVICE_ERROR",
        )

