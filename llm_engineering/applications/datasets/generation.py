from abc import ABC, abstractmethod
from langchain_core.prompts import PromptTemplate
from llm_engineering.applications.networks.dsl_generator import DSLGenerator

from llm_engineering.domains.documents import Document
from llm_engineering.domains.prompt import GenerateDatasetSamplesPrompt, Prompt
from llm_engineering.domains.dataset import InstructDataset, TrainTestSplit
from . import utils as generation_utils

class DatasetGeneration(ABC):
    system_prompt_template = """You are a helpful assistant who generates {dataset_format} based on the given context. \
        Provide your response in JSON format.
    """

    prompt_template_str: str | None = None

    dsl_generator = DSLGenerator()

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
            input_variables=input_variables,
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

        prompt = GenerateDatasetSamplesPrompt(
            template=prompt_template.template,
            input_variables=input_variables,
            content=prompt,
            document=document,
        )

        return prompt

    @classmethod
    def generate(
        cls,
        prompts: list[GenerateDatasetSamplesPrompt],
        test_size: float = 0.2,
    ) -> TrainTestSplit:

        dataset = cls.dsl_generator(cls.get_system_prompt().content, prompts)

        processed_datasets = cls.post_process_datasets(dataset, test_size=test_size)

        return processed_datasets

    @classmethod
    @abstractmethod
    def post_process_datasets(
        cls, dataset: InstructDataset, test_size: float
    ) -> TrainTestSplit:
        pass

class InstructiveDatasetGenerator(DatasetGeneration):
    prompt_template_str = """Based on the following few-shot examples, convert the Vietnamese geometry problem to GMBL format.

IMPORTANT:
- Output MUST be a valid JSON array with ONE object
- Follow EXACTLY the patterns in examples
- Use ONLY keywords from examples
- If unsure, output simpler GMBL

FEW-SHOT EXAMPLES:
[
  {
    "instruction": "Tam giác ABC, góc ABC = 90, điểm M là trung điểm của đoạn thẳng BC",
    "answer": "(param (A B C) triangle)\\n(assert (right-angle A B C))\\n(define M point (midpoint B C))"
  },
  {
    "instruction": "Tam giác ABC, AB = AC",
    "answer": "(param (A B C) triangle)\\n(assert (congruent-segments A B A C))"
  },
  {
    "instruction": "Đường tròn O",
    "answer": "(param O circle)"
  },
  {
    "instruction": "Tam giác ABC, góc ABC = 90, góc BAC = 60, góc ACB = 30",
    "answer": "(param (A B C) triangle)\\n(assert (right-angle A B C))\\n(assert (angle B A C 60))\\n(assert (angle A C B 30))"
  },
  {
    "instruction": "Hình vuông ABCD, đường chéo AC",
    "answer": "(param (A B C D) square)\\n(define AC line (line A C))"
  }
]

Problem: {{extract}}

Output JSON array with ONE object:"""



    @classmethod
    def post_process_datasets(cls, dataset: InstructDataset, test_size: float) -> TrainTestSplit:
        train_test_split = generation_utils.create_instruct_train_test_split(
            [dataset], test_size=test_size, random_state=42
        )

        return train_test_split

