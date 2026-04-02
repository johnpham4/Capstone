"""Shared domain utilities - Common types and exceptions."""

from .exceptions import (
    ImproperlyConfigured,
    DomainError,
)
from .types import DataCategory

__all__ = [
    # Exceptions
    "ImproperlyConfigured",
    "DomainError",
    # Types
    "DataCategory",
]

