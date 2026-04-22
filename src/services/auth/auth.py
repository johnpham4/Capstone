import asyncio
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
import hashlib
import hmac
import re
import secrets
import smtplib

import jwt
from google.auth.transport.requests import Request as GoogleRequest
from google.oauth2 import id_token as google_id_token
from sqlalchemy.ext.asyncio import AsyncSession

from src.config.settings import settings
from src.core.security import create_access_token, create_refresh_token, get_password_hash, verify_password
from src.infrastructures.redis.cache import email_otp_store, token_blacklist
from src.models.dto.auth import Token
from src.models.dto.user import User, UserInDB, UserCreate
from src.repositories.user import UserRepository


class AuthRateLimitError(ValueError):
    def __init__(self, message: str, retry_after_seconds: int | None = None):
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class AuthDeliveryError(RuntimeError):
    pass


class AuthService:

    def __init__(self, db: AsyncSession) -> None:
        self._repo = UserRepository(db)

    async def login(self, username: str, password: str) -> tuple[Token, str]:
        user = await self.authenticate(username, password)
        if not user:
            raise ValueError("Incorrect username or password")

        if not bool(user.email_verified):
            raise ValueError("Email is not verified. Please verify OTP before signing in.")

        return self._issue_tokens(user.username)

    async def request_email_otp(self, email: str) -> None:
        email_normalized = self._normalize_email(email)

        requests_count = await email_otp_store.mark_request(
            email=email_normalized,
            window_seconds=settings.OTP_REQUEST_WINDOW_SECONDS,
        )
        if requests_count > settings.OTP_MAX_REQUESTS_PER_WINDOW:
            raise AuthRateLimitError(
                "Too many OTP requests. Please try again later.",
                retry_after_seconds=settings.OTP_REQUEST_WINDOW_SECONDS,
            )

        cooldown_seconds = await email_otp_store.get_cooldown_seconds(email_normalized)
        if cooldown_seconds > 0:
            raise AuthRateLimitError(
                "Please wait before requesting another OTP.",
                retry_after_seconds=cooldown_seconds,
            )

        code = f"{secrets.randbelow(1_000_000):06d}"
        otp_hash = self._hash_otp(email_normalized, code)

        await email_otp_store.save_otp_hash(
            email=email_normalized,
            otp_hash=otp_hash,
            ttl_seconds=settings.OTP_TTL_SECONDS,
        )

        try:
            await asyncio.to_thread(self._send_otp_email, email_normalized, code)
        except Exception as exc:
            await email_otp_store.delete_otp(email_normalized)
            raise AuthDeliveryError("OTP delivery failed. Please try again later.") from exc

        await email_otp_store.set_cooldown(
            email=email_normalized,
            ttl_seconds=settings.OTP_RESEND_COOLDOWN_SECONDS,
        )

    async def verify_email_otp(self, email: str, code: str) -> tuple[Token, str]:
        email_normalized = self._normalize_email(email)
        code_normalized = self._normalize_otp_code(code)

        otp_hash, attempts = await email_otp_store.get_otp_hash_and_attempts(email_normalized)
        if otp_hash is None:
            raise ValueError("Invalid or expired OTP")

        if attempts >= settings.OTP_MAX_ATTEMPTS:
            await email_otp_store.delete_otp(email_normalized)
            raise ValueError("OTP verification attempts exceeded")

        incoming_hash = self._hash_otp(email_normalized, code_normalized)
        if not hmac.compare_digest(otp_hash, incoming_hash):
            next_attempts = await email_otp_store.increment_attempts(email_normalized)
            if next_attempts >= settings.OTP_MAX_ATTEMPTS:
                await email_otp_store.delete_otp(email_normalized)
            raise ValueError("Invalid or expired OTP")

        await email_otp_store.delete_otp(email_normalized)
        user_model = await self._get_or_create_user_by_email(email_normalized)
        return self._issue_tokens(user_model.username)

    async def login_with_google(self, id_token_str: str) -> tuple[Token, str]:
        if not settings.GOOGLE_CLIENT_ID:
            raise ValueError("GOOGLE_CLIENT_ID is not configured")

        payload = await asyncio.to_thread(
            google_id_token.verify_oauth2_token,
            id_token_str,
            GoogleRequest(),
            settings.GOOGLE_CLIENT_ID,
        )

        email = str(payload.get("email", "")).strip().lower()
        email_verified = bool(payload.get("email_verified", False))
        if not email or not email_verified:
            raise ValueError("Google account email is missing or unverified")

        user_model = await self._repo.get_by_email(email)
        if user_model is None:
            user_model = await self._create_user_from_google_payload(payload, email)
        elif not user_model.email_verified:
            updated = await self._repo.update(user_model.id, {"email_verified": True})
            if updated is not None:
                user_model = updated

        return self._issue_tokens(user_model.username)

    async def _get_or_create_user_by_email(self, email: str):
        user_model = await self._repo.get_by_email(email)
        if user_model is not None:
            if not user_model.email_verified:
                updated = await self._repo.update(user_model.id, {"email_verified": True})
                if updated is not None:
                    user_model = updated
            return user_model

        preferred_username = email.split("@")[0]
        username = await self._ensure_unique_username(preferred_username)
        return await self._repo.create(
            {
                "username": username,
                "email": email,
                "hashed_password": get_password_hash(secrets.token_urlsafe(32)),
                "disabled": False,
                "email_verified": True,
            }
        )

    async def _create_user_from_google_payload(self, payload: dict, email: str):
        preferred_username = str(payload.get("name") or payload.get("given_name") or email.split("@")[0])
        username = await self._ensure_unique_username(preferred_username)

        return await self._repo.create(
            {
                "username": username,
                "email": email,
                "hashed_password": get_password_hash(secrets.token_urlsafe(32)),
                "disabled": False,
                "email_verified": True,
            }
        )

    async def _ensure_unique_username(self, seed: str) -> str:
        base = self._normalize_username(seed)
        if await self._repo.get_by_username(base) is None:
            return base

        suffix_len = 6
        max_base_len = 50 - 1 - suffix_len
        trimmed_base = base[:max_base_len]
        for _ in range(10):
            candidate = f"{trimmed_base}_{secrets.token_hex(3)}"
            if await self._repo.get_by_username(candidate) is None:
                return candidate

        raise ValueError("Could not allocate unique username")

    @staticmethod
    def _normalize_username(seed: str) -> str:
        normalized = re.sub(r"[^a-zA-Z0-9_]+", "_", seed.strip().lower())
        normalized = normalized.strip("_")
        if not normalized:
            normalized = f"user_{secrets.token_hex(3)}"
        return normalized[:50]

    @staticmethod
    def _normalize_email(email: str) -> str:
        normalized = email.strip().lower()
        if not normalized or "@" not in normalized:
            raise ValueError("Invalid email format")
        return normalized

    @staticmethod
    def _normalize_otp_code(code: str) -> str:
        normalized = re.sub(r"\D", "", code)
        if len(normalized) != 6:
            raise ValueError("OTP must be a 6-digit code")
        return normalized

    @staticmethod
    def _hash_otp(email: str, code: str) -> str:
        material = f"{email}:{code}:{settings.OTP_HASH_SECRET}".encode("utf-8")
        return hashlib.sha256(material).hexdigest()

    @staticmethod
    def _send_otp_email(recipient_email: str, code: str) -> None:
        if not settings.SMTP_HOST or not settings.SMTP_FROM_EMAIL:
            raise ValueError("SMTP settings are not configured")

        if settings.SMTP_PORT == 465 and not settings.SMTP_USE_SSL:
            raise ValueError("SMTP_PORT=465 requires SMTP_USE_SSL=true (implicit TLS)")

        msg = EmailMessage()
        msg["Subject"] = "Your GeoUni verification code"
        msg["From"] = settings.SMTP_FROM_EMAIL
        msg["To"] = recipient_email
        msg.set_content(
            "Your verification code is: "
            f"{code}\n\n"
            f"This code expires in {max(settings.OTP_TTL_SECONDS // 60, 1)} minutes. "
            "If you did not request this code, please ignore this email."
        )

        if settings.SMTP_USE_SSL:
            with smtplib.SMTP_SSL(settings.SMTP_HOST, settings.SMTP_PORT, timeout=20) as smtp:
                if settings.SMTP_USERNAME and settings.SMTP_PASSWORD:
                    smtp.login(settings.SMTP_USERNAME, settings.SMTP_PASSWORD)
                smtp.send_message(msg)
            return

        with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=20) as smtp:
            if settings.SMTP_USE_STARTTLS:
                smtp.starttls()
            if settings.SMTP_USERNAME and settings.SMTP_PASSWORD:
                smtp.login(settings.SMTP_USERNAME, settings.SMTP_PASSWORD)
            smtp.send_message(msg)

    async def refresh(self, refresh_token: str) -> tuple[Token, str]:
        try:
            payload = jwt.decode(
                refresh_token,
                settings.JWT_SECRET_KEY,
                algorithms=[settings.JWT_ALGORITHM],
            )
        except jwt.ExpiredSignatureError as exc:
            raise ValueError("Refresh token expired") from exc
        except jwt.PyJWTError as exc:
            raise ValueError("Invalid refresh token") from exc

        if payload.get("typ") != "refresh":
            raise ValueError("Invalid token type")

        username: str | None = payload.get("sub")
        jti: str | None = payload.get("jti")
        exp: int | None = payload.get("exp")
        if not username or not jti or not exp:
            raise ValueError("Invalid refresh token payload")

        if await token_blacklist.is_revoked(jti):
            raise ValueError("Refresh token has been revoked")

        # Rotate refresh token: invalidate old one then issue a fresh pair.
        await token_blacklist.revoke(jti, self._ttl_from_exp(exp))
        return self._issue_tokens(username)

    def _issue_tokens(self, username: str) -> tuple[Token, str]:
        access_token = create_access_token(
            data={"sub": username},
            expires_delta=timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES),
        )
        refresh_token = create_refresh_token(
            data={"sub": username},
            expires_delta=timedelta(days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS),
        )
        return Token(access_token=access_token, token_type="bearer"), refresh_token

    async def register(self, data: UserCreate) -> User:
        normalized_email = data.email.strip().lower()

        existing = await self._repo.get_by_username(data.username)
        if existing:
            raise ValueError("Username already registered")

        existing_email = await self._repo.get_by_email(normalized_email)
        if existing_email:
            raise ValueError("Email already registered")

        user_model = await self._repo.create(
            {
                "username": data.username,
                "email": normalized_email,
                "hashed_password": get_password_hash(data.password),
                "disabled": False,
                "email_verified": False,
            }
        )

        return User(
            id=user_model.id,
            username=user_model.username,
            email=user_model.email,
            disabled=user_model.disabled,
            email_verified=user_model.email_verified,
        )

    async def logout(self, token: str | None = None, refresh_token: str | None = None) -> None:
        if token:
            await self._revoke_token(token, expected_type="access")
        if refresh_token:
            await self._revoke_token(refresh_token, expected_type="refresh")

    async def _revoke_token(self, token: str, expected_type: str | None = None) -> None:
        try:
            payload = jwt.decode(
                token,
                settings.JWT_SECRET_KEY,
                algorithms=[settings.JWT_ALGORITHM],
            )
            token_type: str | None = payload.get("typ")
            if expected_type and token_type != expected_type:
                return

            jti: str | None = payload.get("jti")
            exp: int | None = payload.get("exp")
            if jti and exp:
                await token_blacklist.revoke(jti, self._ttl_from_exp(exp))
        except jwt.PyJWTError:
            # Token Ä‘Ã£ invalid/háº¿t háº¡n â†’ khÃ´ng cáº§n blacklist
            pass

    @staticmethod
    def _ttl_from_exp(exp: int) -> int:
        return int(exp - datetime.now(timezone.utc).timestamp())

    @staticmethod
    def _ttl_from_exp(exp: int) -> int:
        return int(exp - datetime.now(timezone.utc).timestamp())

    async def authenticate(self, identifier: str, password: str) -> UserInDB | None:
        normalized = identifier.strip()
        user = await self.get_user_by_username(normalized)
        if user is None and "@" in normalized:
            user = await self.get_user_by_email(normalized)

        # Also allow email lookup fallback when identifier is not an exact username.
        if user is None:
            user = await self.get_user_by_email(normalized)

        if user is None:
            return None
        if not verify_password(password, user.hashed_password):
            return None
        return user

    async def get_user_by_username(self, username: str) -> UserInDB | None:
        user_model = await self._repo.get_by_username(username)
        if user_model is None:
            return None
        return self._repo.to_schema(user_model)

    async def get_user_by_email(self, email: str) -> UserInDB | None:
        user_model = await self._repo.get_by_email(email.strip().lower())
        if user_model is None:
            return None
        return self._repo.to_schema(user_model)
