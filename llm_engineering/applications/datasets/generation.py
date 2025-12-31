from abc import ABC, abstractmethod

import tiktoken
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from loguru import logger

from llm_engineering.domains.documents import Document
from llm_engineering.domains.prompt import GenerateDatasetSamplesPrompt, Prompt
from llm_engineering.domains.types import DataCategory
from llm_engineering import settings
from llm_engineering.applications.datasets.output_parser import ListPydanticOutputParser
from llm_engineering.domains.dataset import InstructDataset, InstructDatasetSample, TrainTestSplit
from llm_engineering.application import utils
from . import utils as generation_utils

class DatasetGeneration(ABC):
    tokenizer = tiktoken.encoding_for_model(settings.OPENAI_MODEL_ID)

    system_prompt_template = """You are a helpful assistant who generates {dataset_format} based on the given context. \
        Provide your response in JSON format.
    """

    prompt_template_str: str | None = None

    @classmethod
    def get_system_prompt(cls) -> Prompt:
        dataset_format = (
            "instruction-answer pairs"
        )
        input_variables = {
            "dataset_format": dataset_format
        }
        system_prompt = cls.system_prompt_template.format(**input_variables)

        return Prompt(
            template=cls.system_prompt_template,
            input_variables="input_variables",
            content=system_prompt
        )

    @classmethod
    def get_prompt(cls, document: Document) -> GenerateDatasetSamplesPrompt:
        prompt_template = PromptTemplate.from_template(
            template=cls.prompt_template_str,
            template_format="jinja2",
        )

        input_variables = {
            "extract": document.caption_vn
        }
        prompt = prompt_template.format(**input_variables)
        prompt_tokens = cls.tokenizer.encode(prompt)
        if len(prompt_tokens) > settings.OPENAI_MAX_TOKEN_WINDOW:
            prompt_tokens = prompt_tokens[: settings.OPENAI_MAX_TOKEN_WINDOW]
            prompt = cls.tokenizer.decode(prompt_tokens)

        prompt = GenerateDatasetSamplesPrompt(
            template=prompt_template.template,
            input_variables=input_variables,
            content=prompt,
            num_tokens=len(prompt_tokens),
            document=document,
        )

        return prompt

    @classmethod
    def generate(
        cls,
        prompts: list[GenerateDatasetSamplesPrompt],
        test_size: float = 0.2,
    ) -> TrainTestSplit:

        def _to_langchain(
            prompt: GenerateDatasetSamplesPrompt
        ) -> list[BaseMessage]:
            messages = [
                SystemMessage(content=cls.get_system_prompt().content),
                HumanMessage(content=prompt.content),
            ]

            return messages

        assert settings.OPENAI_API_KEY is not None, "OpenAI API key must be set to generate datasets"

        llm = ChatOpenAI(
            model=settings.OPENAI_MODEL_ID,
            api_key=settings.OPENAI_API_KEY,
            max_tokens=1200,
            temperature=0.7,
        )

        parser = ListPydanticOutputParser(pydantic_object=InstructDatasetSample)

        chain = llm | parser

        datasets = {}
        for category, category_prompts in prompts.items():
            langchain_category_prompts = [_to_langchain(prompt) for prompt in category_prompts]
            batches = utils.misc.batch(langchain_category_prompts, size=24)

            flattened_instruct_dataset_samples = []
            for batch in batches:
                try:
                    batched_dataset_samples = chain.batch(batch, stop=None)

                    for instruct_dataset_sample_batch in batched_dataset_samples:
                        flattened_instruct_dataset_samples.extend(instruct_dataset_sample_batch)
                except OutputParserException:
                    logger.exception(f"Failed to parse the output JSON for a batch for category {category}")

            dataset = InstructDataset(category=category, samples=flattened_instruct_dataset_samples)
            datasets[category] = dataset
            logger.info(f"Generated {len(dataset.samples)} samples for category '{category}'.")

        processed_datasets = cls.post_process_datasets(datasets, test_size=test_size)

        return processed_datasets

    @classmethod
    @abstractmethod
    def post_process_datasets(
        cls, datasets: dict[DataCategory, InstructDataset], test_size: float
    ) -> TrainTestSplit:
        pass

class InstructiveDatasetGenerator(DatasetGeneration):
    prompt_template_str = f"""
    Based on the following few-shot examples, generate GMBL (Geo Model Building Language),
    a strict domain-specific language for constructing geometry diagrams
    from Vietnamese geometry problems.

    IMPORTANT CONSTRAINTS:
    - You MUST strictly follow the patterns in the examples.
    - You MUST NOT invent, hallucinate, or create any new keywords, operators, or structures.
    - Use ONLY the vocabulary, syntax, and structures appearing in the examples.
    - If some information is missing, OMIT it. DO NOT guess.
    - Do NOT explain. Do NOT add comments.

    OUTPUT FORMAT:
    - Output MUST be a valid JSON array.
    - Output MUST be directly parseable by Python using json.loads().
    - Output ONLY JSON. No extra text.

    Each JSON object MUST have the structure:
    {{
    "instruction": "<Vietnamese geometry problem>",
    "answer": "<GMBL program>"
    }}

    FEW-SHOT EXAMPLES:
    [
    {{
        "instruction": "Tam giác ABC, góc ABC = 90, điểm M là trung điểm của đoạn thẳng BC",
        "answer": "(param (A B C) triangle)\n(assert (right-angle A B C))\n(define M point (midpoint B C))"
    }},
    {{
        "instruction": "Tam giác ABC, AB = AC",
        "answer": "(param (A B C) triangle)\n(assert (congruent-segments A B A C))"
    }},
    {{
        "instruction": "Hình vuông ABCD, hai đường chéo AC và BD cắt nhau tại O",
        "answer": "(param (A B C D) quadrilateral)\n(define O point (intersection (line A C) (line B D)))"
    }},
    {{
        "instruction": "Tam giác ABC, góc ABC = 90, điểm D nằm trên đoạn thẳng AB",
        "answer": "(param (A B C) triangle)\n(assert (right-angle A B C))\n(define D point (on-segment A B))"
    }},
    {{
        "instruction": "Tam giác ABC, góc ABC = 90, góc BAC = 45, góc ACB = 45",
        "answer": "(param (A B C) triangle)\n(assert (right-angle A B C))\n(assert (angle B A C 45))\n(assert (angle A C B 45))"
    }}
    ]

    Now generate the GMBL output for the following problem:
    {{extract}}
    """



    @classmethod
    def post_process_datasets(cls, datasets: dict[InstructDataset], test_size: float) -> TrainTestSplit:
        train_test_split = generation_utils.create_instruct_train_test_split(
            datasets, test_size=test_size, random_state=42
        )

        return train_test_split

