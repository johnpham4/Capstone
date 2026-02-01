"""Prompt domain models."""

from pydantic import BaseModel
from typing import Optional

from src.models.domain.training.documents import Document


class Prompt(BaseModel):
    """Prompt template for LLM generation.

    Uses BaseModel for validation.
    """
    template: str
    input_variables: dict
    content: str
    num_tokens: Optional[int] = None


class GenerateDatasetSamplesPrompt(Prompt):
    """Prompt for generating dataset samples from documents."""
    document: Document

    class Config:
        arbitrary_types_allowed = True
