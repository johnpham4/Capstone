import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.infrastructures.database.base import Base


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


def _generate_uuid() -> str:
    return str(uuid.uuid4())


class UserModel(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_generate_uuid
    )
    username: Mapped[str] = mapped_column(
        String(50), unique=True, index=True, nullable=False
    )
    email: Mapped[str] = mapped_column(String(255), nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    disabled: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    requests: Mapped[list["RequestModel"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class RequestModel(Base, TimestampMixin):
    __tablename__ = "requests"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_generate_uuid
    )
    user_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    input_text: Mapped[str] = mapped_column(Text, nullable=False)
    mode: Mapped[str] = mapped_column(String(20), nullable=False, default="auto")  # auto / diagram / solve / both
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")  # pending / processing / completed / failed
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Relationships
    user: Mapped["UserModel | None"] = relationship(back_populates="requests")
    diagram: Mapped["DiagramModel | None"] = relationship(
        back_populates="request", cascade="all, delete-orphan", uselist=False
    )
    solution: Mapped["SolutionModel | None"] = relationship(
        back_populates="request", cascade="all, delete-orphan", uselist=False
    )


class DiagramModel(Base, TimestampMixin):
    __tablename__ = "diagrams"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_generate_uuid
    )
    request_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("requests.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    dsl: Mapped[str] = mapped_column(Text, nullable=False)
    image_base64: Mapped[str | None] = mapped_column(Text, nullable=True)
    generation_time_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    render_time_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    request: Mapped["RequestModel"] = relationship(back_populates="diagram")


class SolutionModel(Base, TimestampMixin):
    __tablename__ = "solutions"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_generate_uuid
    )
    request_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("requests.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    request: Mapped["RequestModel"] = relationship(back_populates="solution")


class RegistryModel(Base, TimestampMixin):
    __tablename__ = "registry"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_generate_uuid
    )
    name_hf: Mapped[str] = mapped_column()
    version: Mapped[int] = mapped_column()
    alias: Mapped[str] = mapped_column()
    prompt: Mapped[str] = mapped_column()

