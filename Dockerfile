# ─── Base stage ───────────────────────────────────────────────
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# ─── Dependencies stage ──────────────────────────────────────
FROM base AS dependencies

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# ─── Runtime stage ────────────────────────────────────────────
FROM base AS runtime

COPY --from=dependencies /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

COPY . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ─── API target ───────────────────────────────────────────────
FROM runtime AS api
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

# ─── Celery worker target ────────────────────────────────────
FROM runtime AS worker
CMD ["celery", "-A", "src.infrastructures.celery.config", "worker", \
     "--loglevel=info", "--concurrency=4"]

# ─── Flower monitoring target ────────────────────────────────
FROM runtime AS flower
EXPOSE 5555
CMD ["celery", "-A", "src.infrastructures.celery.config", "flower", \
     "--port=5555"]
