from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from llm_engineering.applications.utils import misc
from loguru import logger

from llm_engineering.applications.datasets.output_parser import ListPydanticOutputParser
from llm_engineering.applications.networks.base import SingletonMeta
from llm_engineering.domains.dataset import InstructDataset, InstructDatasetSample
from llm_engineering.domains.prompt import GenerateDatasetSamplesPrompt
from llm_engineering.applications.networks.qwen7B import QwenLocalLLM

class DSLGenerator(metaclass=SingletonMeta):

    def __init__(self):
        self.llm = QwenLocalLLM()

    def __call__(
        self,
        prompts: list[GenerateDatasetSamplesPrompt]
    ) -> InstructDataset:

        def _to_prompt(prompt: GenerateDatasetSamplesPrompt) -> str:
            messages = [
                SystemMessage(content=self.get_system_prompt().content),
                HumanMessage(content=prompt.content),
            ]
            return "\n".join(m.content for m in messages)

        parser = ListPydanticOutputParser(
            pydantic_object=InstructDatasetSample
        )

        chain = self.llm | parser

        string_prompts = [_to_prompt(p) for p in prompts]
        batches = misc.batch(string_prompts, size=24)

        flattened_samples = []

        for batch in batches:
            try:
                results = chain.batch(batch)
                for r in results:
                    flattened_samples.extend(r)
            except OutputParserException:
                logger.exception("Failed to parse output JSON")

        dataset = InstructDataset(samples=flattened_samples)
        logger.info(f"Generated {len(dataset.samples)} samples total.")
        return dataset
