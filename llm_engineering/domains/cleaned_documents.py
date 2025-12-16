from abc import ABC
from typing import Optional, List
from pydantic import Field

from llm_engineering.domains.orm.nosql import NoSQLBaseDocument
from llm_engineering.domains.types import DataCategory, Language

class CleanedDocuments(NoSQLBaseDocument, ABC):
    document_id: str
    problem: str
    diagram: Optional[str] = Field(default=None, description="Diagram image path or base64")
    platform: DataCategory


class CleanedTextbookDocuments(CleanedDocuments):
    page_number: Optional[int]
    link: Optional[int]

    class Settings:
        name = "cleaned_textbook"
