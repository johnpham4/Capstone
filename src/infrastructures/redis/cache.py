import hashlib
import json
from typing import Optional

import redis
from loguru import logger

from src.config.settings.base import settings


class RenderCache:
    _PREFIX = "render:cache:"
    _DEFAULT_TTL = 60 * 60 * 24 * 7  # 7 days — images are deterministic

    def __init__(self, ttl: int = _DEFAULT_TTL):
        self.ttl = ttl
        try:
            self._client = redis.Redis.from_url(
                settings.REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=3,
            )
            self._client.ping()
            self._available = True
            logger.info("Render cache connected to Redis")
        except (redis.ConnectionError, redis.TimeoutError) as exc:
            logger.warning(f"Redis unavailable – render cache disabled: {exc}")
            self._available = False

    def get(self, dsl: str, **kwargs) -> Optional[str]:
        if not self._available:
            return None
        key = self._make_key(dsl, **kwargs)
        try:
            cached = self._client.get(key)
            if cached is not None:
                logger.debug(f"Render cache HIT  [{key[:40]}…]")
                return cached
            logger.debug(f"Render cache MISS [{key[:40]}…]")
            return None
        except redis.RedisError as exc:
            logger.warning(f"Redis read error – skipping cache: {exc}")
            return None

    def set(self, dsl: str, image_base64: str, **kwargs) -> None:
        """Store rendered image_base64 keyed by DSL."""
        if not self._available:
            return
        key = self._make_key(dsl, **kwargs)
        try:
            self._client.setex(key, self.ttl, image_base64)
            logger.debug(f"Render cache SET  [{key[:40]}…]  ttl={self.ttl}s")
        except redis.RedisError as exc:
            logger.warning(f"Redis write error – image not cached: {exc}")

    def invalidate(self, dsl: str, **kwargs) -> None:
        if not self._available:
            return
        key = self._make_key(dsl, **kwargs)
        try:
            self._client.delete(key)
        except redis.RedisError:
            pass

    def flush_all(self) -> int:
        """Delete all render cache entries.  Returns count deleted."""
        if not self._available:
            return 0
        try:
            keys = self._client.keys(f"{self._PREFIX}*")
            if keys:
                return self._client.delete(*keys)
            return 0
        except redis.RedisError:
            return 0

    @classmethod
    def _make_key(cls, dsl: str, **kwargs) -> str:
        """Deterministic cache key: hash(normalised_dsl + render params)."""
        normalised = dsl.strip()
        raw = json.dumps({"dsl": normalised, **kwargs}, sort_keys=True)
        digest = hashlib.sha256(raw.encode()).hexdigest()[:32]
        return f"{cls._PREFIX}{digest}"


# Singleton
render_cache = RenderCache()
