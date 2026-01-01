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
    prompt_template_str = """Convert this Vietnamese geometry problem to GMBL format.

RULES:
1. Return ONLY a JSON array with ONE object
2. JSON must have "instruction" and "answer" fields
3. Use \\n for newlines in "answer" field
4. NO markdown, NO extra text, NO explanations
5. Follow these examples:

Example 1:
[{"instruction": "Tam giác ABC, góc ABC = 90, điểm M là trung điểm của đoạn thẳng BC", "answer": "(param (A B C) triangle)\\n(assert (right-angle A B C))\\n(define M point (midpoint B C))"}]

Example 2:
[{"instruction": "Tam giác ABC, AB = AC", "answer": "(param (A B C) triangle)\\n(assert (congruent-segments A B A C))"}]

Example 3:
[{"instruction": "Đường tròn O", "answer": "(param O circle)"}]

Now convert this problem:
{{extract}}

JSON output:"""

    @classmethod
    def post_process_datasets(cls, dataset: InstructDataset, test_size: float) -> TrainTestSplit:
        # Đây là hàm cắt train/test chuẩn
        return generation_utils.create_instruct_train_test_split([dataset], test_size=test_size, random_state=42)
