from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.api.endpoints import register_routes
from src.api.error_handler import register_error_handlers
from src.config.settings.settings import settings
from src.infrastructures.database.session import init_db
from src.infrastructures.redis.connection import RedisConnector


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.INIT_DB_ON_STARTUP:
        logger.info("Initializing database...")
        await init_db()
        logger.info("Database initialized successfully")
    yield
    await RedisConnector.close()
    logger.info("Application shutdown")

app = FastAPI(
    title="GeoUni Backend API",
    description="Geometry problem solving platform with AI-powered agents",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["infra"])
async def health_check():
    return {"status": "ok"}


# Register all API routes
register_routes(app)
register_error_handlers(app)



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )