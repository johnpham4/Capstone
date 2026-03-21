from loguru import logger
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.config.settings.base import settings
from src.models.dto.orchestration import RewriteResponse
from src.prompts.rewriter import REWRITER_SYSTEM_PROMPT


class RewriterAgent:
    def __init__(self) -> None:
        self.llm = ChatOpenAI(
            model=settings.OPENAI_MODEL_ID,
            api_key=settings.OPENAI_API_KEY,
            temperature=0,
            max_tokens=256,
            timeout=30,
        )
        self.parser = PydanticOutputParser(pydantic_object=RewriteResponse)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", REWRITER_SYSTEM_PROMPT),
            ("human", "{user_input}"),
        ]).partial(format_instructions=self.parser.get_format_instructions())
        self.chain = self.prompt | self.llm | self.parser

    def execute(self, user_input: str) -> RewriteResponse:
        user_input = (user_input or "").strip()
        try:
            return self.chain.invoke({"user_input": user_input})
        except Exception as e:
            logger.warning(f"RewriterAgent LLM failed, using fallback: {e}")
            return RewriteResponse(problem_statement=user_input, mode="diagram")

