import redis.asyncio as aioredis
from loguru import logger

from src.config.settings import settings


class RedisConnector:
    _client: aioredis.Redis | None = None

    @classmethod
    async def get(cls) -> aioredis.Redis | None:
        if cls._client is not None:
            return cls._client
        try:
            cls._client = aioredis.from_url(
                settings.REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=3,
                max_connections=20,
            )
            await cls._client.ping()
            logger.info("Async Redis pool connected")
            return cls._client
        except (aioredis.ConnectionError, aioredis.TimeoutError) as exc:
            logger.warning(f"Redis unavailable: {exc}")
            cls._client = None
            return None

    @classmethod
    async def close(cls) -> None:
        if cls._client is not None:
            await cls._client.aclose()
            cls._client = None
            logger.info("Redis connection closed")
