import redis.asyncio as aioredis
import hashlib
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


class EmailOtpStore:
    _OTP_PREFIX = "otp:email:data:"
    _COOLDOWN_PREFIX = "otp:email:cooldown:"
    _REQUEST_PREFIX = "otp:email:req:"

    @staticmethod
    async def _client() -> aioredis.Redis | None:
        return await RedisConnector.get()

    @staticmethod
    def _email_key(email: str) -> str:
        normalized = email.strip().lower()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    async def mark_request(self, email: str, window_seconds: int) -> int:
        client = await self._client()
        if client is None:
            return 1
        key = f"{self._REQUEST_PREFIX}{self._email_key(email)}"
        try:
            current = await client.incr(key)
            if current == 1:
                await client.expire(key, window_seconds)
            return int(current)
        except aioredis.RedisError as exc:
            logger.warning(f"Redis write error – OTP request counter fallback: {exc}")
            return 1

    async def set_cooldown(self, email: str, ttl_seconds: int) -> None:
        client = await self._client()
        if client is None or ttl_seconds <= 0:
            return
        key = f"{self._COOLDOWN_PREFIX}{self._email_key(email)}"
        try:
            await client.setex(key, ttl_seconds, "1")
        except aioredis.RedisError as exc:
            logger.warning(f"Redis write error – OTP cooldown not set: {exc}")

    async def get_cooldown_seconds(self, email: str) -> int:
        client = await self._client()
        if client is None:
            return 0
        key = f"{self._COOLDOWN_PREFIX}{self._email_key(email)}"
        try:
            ttl = await client.ttl(key)
            return max(int(ttl), 0)
        except aioredis.RedisError as exc:
            logger.warning(f"Redis read error – OTP cooldown fallback: {exc}")
            return 0

    async def save_otp_hash(self, email: str, otp_hash: str, ttl_seconds: int) -> None:
        client = await self._client()
        if client is None:
            return
        key = f"{self._OTP_PREFIX}{self._email_key(email)}"
        try:
            await client.hset(key, mapping={"otp_hash": otp_hash, "attempts": "0"})
            await client.expire(key, ttl_seconds)
        except aioredis.RedisError as exc:
            logger.warning(f"Redis write error – OTP record not saved: {exc}")

    async def get_otp_hash_and_attempts(self, email: str) -> tuple[str | None, int]:
        client = await self._client()
        if client is None:
            return None, 0
        key = f"{self._OTP_PREFIX}{self._email_key(email)}"
        try:
            payload = await client.hgetall(key)
            if not payload:
                return None, 0
            otp_hash = payload.get("otp_hash")
            attempts_raw = payload.get("attempts", "0")
            return otp_hash, int(attempts_raw)
        except (aioredis.RedisError, ValueError) as exc:
            logger.warning(f"Redis read error – OTP lookup fallback: {exc}")
            return None, 0

    async def increment_attempts(self, email: str) -> int:
        client = await self._client()
        if client is None:
            return 0
        key = f"{self._OTP_PREFIX}{self._email_key(email)}"
        try:
            return int(await client.hincrby(key, "attempts", 1))
        except aioredis.RedisError as exc:
            logger.warning(f"Redis write error – OTP attempts fallback: {exc}")
            return 0

    async def delete_otp(self, email: str) -> None:
        client = await self._client()
        if client is None:
            return
        key = f"{self._OTP_PREFIX}{self._email_key(email)}"
        try:
            await client.delete(key)
        except aioredis.RedisError as exc:
            logger.warning(f"Redis delete error – OTP cleanup failed: {exc}")


# Singleton
token_blacklist = TokenBlacklist()
email_otp_store = EmailOtpStore()
