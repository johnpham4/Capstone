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
    prompt_template_str = """Convert Vietnamese geometry problem to GMBL (Geometry Meaning-Based Language) following EXACT syntax from examples.

=== GMBL SYNTAX RULES ===

1. DECLARATION (must come first):
   (param A point)              ; Single point
   (param (A B C) triangle)     ; Triangle with 3 points
   (param O circle)             ; Circle
   (param D point (on-seg A B)) ; Point on segment with constraint

2. DEFINITION (derived objects):
   (define O point (circumcenter A B C))  ; Circumcenter
   (define M point (midp A B))            ; Midpoint
   (define l line (line A B))             ; Line through 2 points

3. PREDICATES:
   (on-seg D A B)          ; D on segment AB
   (on-circ P O)           ; P on circle O
   (cong A B C D)          ; AB = CD
   (para l1 l2)            ; l1 parallel l2
   (perp l1 l2)            ; l1 perpendicular l2

4. CRITICAL ERRORS TO AVOID:
   (intersect l1 l2 (param P point))  ; NEVER declare param inside expression
   (define O circle)                  ; If O is center, use "point" not "circle"
   (intersect A B C)                  ; Points cannot intersect
   (passes-through O N)               ; Wrong predicate name

=== CORRECT EXAMPLES ===

Example 1 - Simple triangle:
[{
  "instruction": "Tam giác ABC",
  "answer": "(param (A B C) triangle)"
}]

Example 2 - Triangle with point on segment:
[{
  "instruction": "Tam giác ABC, điểm D nằm trên đoạn thẳng AB",
  "answer": "(param (A B C) triangle)\\n(param D point (on-seg A B))"
}]

Example 3 - Circumcenter:
[{
  "instruction": "Tam giác ABC nhọn, điểm O là tâm đường tròn ngoại tiếp",
  "answer": "(param (A B C) acute-tri)\\n(define O point (circumcenter A B C))"
}]

Example 4 - Midpoint:
[{
  "instruction": "Tam giác ABC, điểm M là trung điểm của BC",
  "answer": "(param (A B C) triangle)\\n(define M point (midp B C))"
}]

Example 5 - Line through points:
[{
  "instruction": "Tam giác ABC, đường thẳng đi qua A và B",
  "answer": "(param (A B C) triangle)\\n(define l line (line A B))"
}]

Example 6 - Right triangle:
[{
  "instruction": "Tam giác ABC vuông tại B",
  "answer": "(param (A B C) triangle)\\n(assert (is-right A B C))"
}]

Example 7 - Circle:
[{
  "instruction": "Đường tròn O, điểm A nằm trên đường tròn",
  "answer": "(param O circle)\\n(param A point (on-circ O))"
}]

=== YOUR TASK ===
Problem: {{extract}}

OUTPUT REQUIREMENTS:
- Return ONLY valid JSON array with ONE object
- Use \\n for newlines in "answer" field
- NO markdown, NO explanation, NO extra text
- Follow EXACT syntax from examples above

JSON output:
"""

    @classmethod
    def post_process_datasets(cls, dataset: InstructDataset, test_size: float) -> TrainTestSplit:

        return generation_utils.create_instruct_train_test_split([dataset], test_size=test_size, random_state=42)