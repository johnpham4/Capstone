
import time

from fastapi import Depends, Request
from loguru import logger
from typing import Annotated

from src.api.dependencies.auth import get_current_user
from src.api.exceptions import RateLimitError
from src.models.dto.user import User


class RateLimiter:
    """Sliding-window rate limiter backed by Redis sorted sets.

    Falls back to no-op if Redis is unavailable (graceful degradation).
    """

    def __init__(self, max_requests: int = 10, window_seconds: int = 60) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds

    async def __call__(
        self,
        request: Request,
        current_user: Annotated[User, Depends(get_current_user)],
    ) -> None:
        """FastAPI Depends entry point — raises RateLimitError if exceeded."""
        try:
            import redis.asyncio as aioredis
            from src.config.settings.base import settings

            client = aioredis.from_url(settings.REDIS_URL, decode_responses=True)

            key = f"rate_limit:{current_user.username}:{request.url.path}"
            now = time.time()
            window_start = now - self.window_seconds

            pipe = client.pipeline()
            # Remove expired entries
            pipe.zremrangebyscore(key, 0, window_start)
            # Count remaining entries in the window
            pipe.zcard(key)
            # Add current request
            pipe.zadd(key, {str(now): now})
            # Set TTL so keys auto-expire
            pipe.expire(key, self.window_seconds + 1)
            results = await pipe.execute()

            request_count = results[1]  # zcard result

            await client.aclose()

            if request_count >= self.max_requests:
                raise RateLimitError(
                    f"Rate limit exceeded: {self.max_requests} requests per "
                    f"{self.window_seconds}s. Try again later."
                )

        except RateLimitError:
            raise
        except Exception as e:
            # Redis down → degrade gracefully, don't block requests
            logger.warning(f"Rate limiter unavailable (Redis error): {e}")


rate_limit_orchestration = RateLimiter(max_requests=10, window_seconds=60)
rate_limit_diagram = RateLimiter(max_requests=20, window_seconds=60)
rate_limit_default = RateLimiter(max_requests=60, window_seconds=60)
