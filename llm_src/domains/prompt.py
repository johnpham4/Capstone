from llm_src.domains.documents import Document
from llm_src.domains.odm.nosql import NoSQLBaseDocument
from llm_src.domains.types import DataCategory


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