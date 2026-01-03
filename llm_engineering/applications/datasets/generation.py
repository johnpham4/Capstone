from abc import ABC, abstractmethod
from llm_engineering.domains.documents import Document
from llm_engineering.domains.prompt import GenerateDatasetSamplesPrompt, Prompt
from llm_engineering.domains.dataset import InstructDataset, TrainTestSplit
from llm_engineering.applications.networks.dsl_generator import DSLGenerator
from . import utils as generation_utils

class DatasetGeneration(ABC):
    system_prompt_template = """You are a geometry formalization system.

You convert Vietnamese geometry problems into GMBL
(Geometry Meaning-Based Language), a formal geometry DSL.

STRICT RULES:
- Do NOT invent geometric objects
- Declare all objects before use
- Use correct predicate arity
- Follow the ontology provided by the user prompt
- Output ONLY valid JSON

Any violation is considered an error.
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
        dataset = cls.dsl_generator(
            cls.get_system_prompt().content,
            prompts,
        )
        processed = cls.post_process_datasets(dataset, test_size=test_size)
        return processed

    @classmethod
    @abstractmethod
    def post_process_datasets(cls, dataset: InstructDataset, test_size: float) -> TrainTestSplit:
        pass


class InstructiveDatasetGenerator(DatasetGeneration):
    prompt_template_str = """Convert the Vietnamese geometry problem into GMBL (Geometry Meaning-Based Language).

GEOMETRY ONTOLOGY:
- ENTITY TYPES: point, line, circle
- DECLARATION: (param A point), (param (A B C) point), (param O circle), (define l line)
- PREDICATES: (passes-through line point), (intersect circle circle point point), (intersect line circle point point), (intersect line line point)

CRITICAL RULES:
1. Declare all objects before use
2. Do NOT invent objects not mentioned in the problem
3. Use correct predicate arity
4. If a line has no name, introduce a fresh symbol like l, l1, l2

OUTPUT FORMAT:
Return ONLY valid JSON array with ONE object:
{
  "instruction": "<original Vietnamese problem>",
  "answer": "<GMBL code with \\n for newlines>"
}

NO markdown code blocks, NO extra text, NO explanations.

EXAMPLES:

[{
  "instruction": "Đường tròn O",
  "answer": "(param O circle)"
}]

[{
  "instruction": "Hai đường tròn O và B cắt nhau tại C và D",
  "answer": "(param (O B) circle)\\n(param (C D) point)\\n(intersect O B C D)"
}]

[{
  "instruction": "Đường thẳng đi qua điểm A",
  "answer": "(param A point)\\n(define l line)\\n(passes-through l A)"
}]

PROBLEM TO CONVERT:
{{extract}}

JSON output:
"""

    @classmethod
    def post_process_datasets(cls, dataset: InstructDataset, test_size: float) -> TrainTestSplit:

        return generation_utils.create_instruct_train_test_split([dataset], test_size=test_size, random_state=42)
