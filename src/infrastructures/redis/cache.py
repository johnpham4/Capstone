import redis.asyncio as aioredis
from loguru import logger

from src.infrastructures.redis.connection import RedisConnector


class TokenBlacklist:
    _PREFIX = "token:blacklist:"

    @staticmethod
    async def _client() -> aioredis.Redis | None:
        return await RedisConnector.get()

    async def revoke(self, jti: str, ttl_seconds: int) -> None:
        client = await self._client()
        if client is None or ttl_seconds <= 0:
            return
        try:
            await client.setex(f"{self._PREFIX}{jti}", ttl_seconds, "1")
            logger.debug(f"Token revoked: jti={jti} ttl={ttl_seconds}s")
        except aioredis.RedisError as exc:
            logger.warning(f"Redis write error – token not blacklisted: {exc}")

    async def is_revoked(self, jti: str) -> bool:
        client = await self._client()
        if client is None:
            return False
        try:
            return await client.exists(f"{self._PREFIX}{jti}") == 1
        except aioredis.RedisError as exc:
            logger.warning(f"Redis read error – assuming token valid: {exc}")
            return False


# Singleton
token_blacklist = TokenBlacklist()
