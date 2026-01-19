from llm_engineering.domains.documents import Document
from llm_engineering.domains.odm.nosql import NoSQLBaseDocument
from llm_engineering.domains.types import DataCategory


class Prompt(NoSQLBaseDocument):
    template: str
    input_variables: dict
    content: str
    num_tokens: int | None = None

    class Config:
        category = DataCategory.PROMPT

class GenerateDatasetSamplesPrompt(Prompt):
    document: Document

    class Config:
        arbitrary_types_allowed = True