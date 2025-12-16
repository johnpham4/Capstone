from abc import ABC
from typing import Optional, List
from pydantic import Field

from llm_engineering.domains.orm.nosql import NoSQLBaseDocument
from llm_engineering.domains.types import DataCategory, Language

class Documents(NoSQLBaseDocument, ABC):
    content: str
    platform: DataCategory


class TextbookDocuments(Documents):
    name: str
    link: Optional[str]

    class Settings:
        name = "textbook"
