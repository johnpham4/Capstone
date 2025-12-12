from abc import ABC
from typing import Optional, List
from pydantic import Field

from llm_engineering.domains.orm.nosql import NoSQLBaseDocument
from llm_engineering.domains.types import DataCategory, Language

class Documents(NoSQLBaseDocument, ABC):
    problem: str
    diagram: Optional[str] = Field(default=None, description="Diagram image path or base64")
    platform: DataCategory


class TextbookPageDocument(NoSQLBaseDocument):
    """1 document = 1 page PDF"""
    page_number: int
    source_pdf: str
    full_text: str = Field(default="", description="All text content on page")
    images: List[str] = Field(default_factory=list, description="List of base64 images")

    class Settings:
        name = "textbook_pages"
