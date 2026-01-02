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
    def generate(cls, prompts: list[GenerateDatasetSamplesPrompt], test_size: float = 0.2, run_id: str = "gmbl_gen") -> TrainTestSplit:
        dataset = cls.dsl_generator(
            cls.get_system_prompt().content,
            prompts,
            run_id=run_id,
            checkpoint_every=50
        )
        processed = cls.post_process_datasets(dataset, test_size=test_size)
        return processed

    @classmethod
    @abstractmethod
    def post_process_datasets(cls, dataset: InstructDataset, test_size: float) -> TrainTestSplit:
        pass


class InstructiveDatasetGenerator(DatasetGeneration):
    prompt_template_str = """You are a geometry formalization system.

Your task is to convert Vietnamese geometry problems into GMBL
(Geometry Meaning-Based Language), a formal geometry DSL.

========================
GEOMETRY ONTOLOGY
========================

ENTITY TYPES:
- point
- line
- circle

DECLARATION:
- (param A point)
- (param (A B C) point)
- (param O circle)
- (param (O B) circle)
- (define l line)

PREDICATES (ARITY FIXED):
- (passes-through line point)
- (intersect circle circle point point)
- (intersect line circle point point)
- (intersect line line point)

RULES:
1. Every object must be declared before use
2. Do NOT invent objects
3. Symbols cannot have multiple types
4. Circles are identified by their centers
5. Intersection predicates MUST use correct arity
6. Use ONE intersection statement per object pair
7. If a line is mentioned without name, introduce a fresh symbol l

========================
OUTPUT FORMAT
========================

Return ONLY a JSON array with ONE object.

The object must contain:
- "instruction"
- "answer"

Use \\n for newlines inside "answer".
NO markdown.
NO explanation.
NO extra text.

========================
EXAMPLES
========================

Example 1:
[{
  "instruction": "Đường tròn O",
  "answer": "(param O circle)"
}]

Example 2:
[{
  "instruction": "Hai đường tròn O và B cắt nhau tại C và D",
  "answer": "(param (O B) circle)\\n(param (C D) point)\\n(intersect O B C D)"
}]

Example 3:
[{
  "instruction": "Đường thẳng đi qua điểm A",
  "answer": "(param A point)\\n(define l line)\\n(passes-through l A)"
}]

========================
PROBLEM
========================

{{extract}}

JSON output:
"""

    @classmethod
    def post_process_datasets(cls, dataset: InstructDataset, test_size: float) -> TrainTestSplit:

        return generation_utils.create_instruct_train_test_split([dataset], test_size=test_size, random_state=42)
