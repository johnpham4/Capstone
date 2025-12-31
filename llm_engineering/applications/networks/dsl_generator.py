from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface.llms import HuggingFacePipeline
import torch
from llm_engineering.applications.utils import misc
from loguru import logger

from llm_engineering.applications.datasets.output_parser import ListPydanticOutputParser
from llm_engineering.applications.networks.base import SingletonMeta
from llm_engineering.domains.dataset import InstructDataset, InstructDatasetSample
from llm_engineering.domains.prompt import GenerateDatasetSamplesPrompt

class DSLGenerator(metaclass=SingletonMeta):

    def __init__(self):
        self.model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"

        self.llm = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_8bit=True
        )

        self.tokenzier = AutoTokenizer.from_pretrained(self.model_name)

        self.pipe = pipeline(
            "text-generation",
            model=self.llm,
            tokenizer=self.tokenzier,
            max_new_tokens=3600,
            temperature=0.2,
        )

        self.hf = HuggingFacePipeline(pipeline=self.pipe)

    def __call__(self, prompts: list[GenerateDatasetSamplesPrompt]) -> list[InstructDataset]:

        def _to_langchain(
            prompt: GenerateDatasetSamplesPrompt
        ) -> list[BaseMessage]:
            messages = [
                SystemMessage(content=self.get_system_prompt().content),
                HumanMessage(content=prompt.content),
            ]

            return messages

        parser = ListPydanticOutputParser(pydantic_object=InstructDatasetSample)

        chain = self.hf | parser

        langchain_prompts = [_to_langchain(prompt) for prompt in prompts]
        batches = misc.batch(langchain_prompts, size=24)

        flattened_instruct_dataset_samples = []
        for batch in batches:
            try:
                batched_dataset_samples = chain.batch(batch, stop=None)

                for instruct_dataset_sample_batch in batched_dataset_samples:
                    flattened_instruct_dataset_samples.extend(instruct_dataset_sample_batch)
            except OutputParserException:
                logger.exception("Failed to parse the output JSON for a batch")

        dataset = InstructDataset(samples=flattened_instruct_dataset_samples)
        logger.info(f"Generated {len(dataset.samples)} samples total.")
        return dataset