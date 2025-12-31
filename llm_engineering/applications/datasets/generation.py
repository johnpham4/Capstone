from abc import ABC, abstractmethod
from llm_engineering.domains.documents import Document
from llm_engineering.domains.prompt import GenerateDatasetSamplesPrompt, Prompt
from llm_engineering.domains.dataset import InstructDataset, TrainTestSplit
from llm_engineering.applications.networks.dsl_generator import DSLGenerator
from . import utils as generation_utils

class DatasetGeneration(ABC):
    system_prompt_template = """You are a helpful assistant who converts Vietnamese geometry problems into GMBL format. \
Provide your response strictly as JSON.
"""

    prompt_template_str: str | None = None

    dsl_generator = DSLGenerator()

    @classmethod
    def get_system_prompt(cls) -> Prompt:
        return Prompt(
            template=cls.system_prompt_template,
            input_variables={},
            content=cls.system_prompt_template
        )

    @classmethod
    def get_prompt(cls, document: Document) -> GenerateDatasetSamplesPrompt:
        from langchain_core.prompts import PromptTemplate

        prompt_template = PromptTemplate.from_template(
            template=cls.prompt_template_str,
            template_format="jinja2",
        )

        input_variables = {"extract": document.caption_vn}

        prompt_text = prompt_template.format(**input_variables)

        return GenerateDatasetSamplesPrompt(
            template=prompt_template.template,
            input_variables=input_variables,
            content=prompt_text,
            document=document
        )

    @classmethod
    def generate(cls, prompts: list[GenerateDatasetSamplesPrompt], test_size: float = 0.2) -> TrainTestSplit:
        dataset = cls.dsl_generator(cls.get_system_prompt().content, prompts)
        processed = cls.post_process_datasets(dataset, test_size=test_size)
        return processed

    @classmethod
    @abstractmethod
    def post_process_datasets(cls, dataset: InstructDataset, test_size: float) -> TrainTestSplit:
        pass


class InstructiveDatasetGenerator(DatasetGeneration):
    prompt_template_str = """Based on the examples below, convert the following Vietnamese geometry problem into GMBL format.

IMPORTANT:
- Output MUST be a JSON array with exactly ONE object.
- Follow the examples exactly.
- Only use the keywords shown in the examples.
- Do NOT add explanations or extra text.
- Escape newlines properly with \\n.

EXAMPLES:
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
    "instruction": "Hình vuông ABCD, đường chéo AC",
    "answer": "(param (A B C D) square)\\n(define AC line (line A C))"
  }
]

Problem: {{extract}}

Output JSON array with ONE object:"""

    @classmethod
    def post_process_datasets(cls, dataset: InstructDataset, test_size: float) -> TrainTestSplit:
        # Đây là hàm cắt train/test chuẩn
        return generation_utils.create_instruct_train_test_split([dataset], test_size=test_size, random_state=42)
