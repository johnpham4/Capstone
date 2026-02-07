from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.api.endpoints import register_routes
from src.config.settings.base import settings

# Create FastAPI application
app = FastAPI(
    title="GeoUni Backend API",
    description="Geometry problem solving platform with AI-powered agents",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (Vercel URL changes on deploy)
    allow_credentials=False,  # Must be False when allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register all API routes
register_routes(app)

logger.info("GeoUni Backend Application initialized")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "GeoUni Backend API",
        "version": "2.0.0",
        "architecture": "Layered Architecture",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health():
    """Global health check endpoint."""
    return {
        "status": "healthy",
        "service": "GeoUni Backend",
        "version": "2.0.0"
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )