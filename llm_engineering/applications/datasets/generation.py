from abc import ABC, abstractmethod

import tiktoken
from langchain_openai import ChatOpenAI
from loguru import logger
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.exceptions import OutputParserException


from llm_engineering.applications.datasets.output_parser import ListPydanticOutputParser
from llm_engineering.applications.utils import misc
from llm_engineering.settings import settings
from llm_engineering.domains.documents import Document
from llm_engineering.domains.prompt import GenerateDatasetSamplesPrompt, Prompt
from llm_engineering.domains.dataset import InstructDataset, InstructDatasetSample, TrainTestSplit
from llm_engineering.applications.networks.dsl_generator import DSLGenerator

from . import utils as generation_utils
from .output_parser import ListPydanticOutputParser

class DatasetGeneration(ABC):
    tokenizer = tiktoken.encoding_for_model(settings.OPENAI_MODEL_ID)

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

    @classmethod
    def get_system_prompt(cls) -> Prompt:
        return Prompt(
            template=cls.system_prompt_template,
            input_variables={},
            content=cls.system_prompt_template
        )

    @classmethod
    def get_prompt(cls, document: Document) -> GenerateDatasetSamplesPrompt:

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
    def generate(cls, prompts: list[GenerateDatasetSamplesPrompt], test_size: float = 0.2, batch_size: int = 16) -> TrainTestSplit:
        def _to_langchain(prompt: GenerateDatasetSamplesPrompt) -> list[BaseMessage]:
            return [
                SystemMessage(content=cls.get_system_prompt().content),
                HumanMessage(content=prompt.content)
            ]

        assert settings.OPENAI_API_KEY is not None, "OpenAI API key must be set to generate datasets"

        llm = ChatOpenAI(
            model=settings.OPENAI_MODEL_ID,
            api_key=settings.OPENAI_API_KEY,
            max_tokens=512,  # Increased for complex GMBL
            temperature=0.3,  # Lower for more deterministic output
        )

        from langchain_core.output_parsers import JsonOutputParser
        parser = JsonOutputParser()
        chain = llm | parser

        messages_batch = [_to_langchain(p) for p in prompts]
        batches = list(misc.batch(messages_batch, size=batch_size))

        samples = []
        for batch_idx, batch in enumerate(batches):
            try:
                raw_outputs = chain.batch(batch, stop=None)

                for idx, raw_output in enumerate(raw_outputs):
                    prompt_idx = batch_idx * batch_size + idx
                    if prompt_idx >= len(prompts):
                        continue

                    prompt = prompts[prompt_idx]

                    # raw_output is either a dict or list
                    if isinstance(raw_output, list):
                        sample_dicts = raw_output
                    elif isinstance(raw_output, dict):
                        sample_dicts = [raw_output]
                    else:
                        logger.warning(f"Unexpected output type: {type(raw_output)}")
                        continue

                    # Inject image_dir into each dict BEFORE Pydantic validation
                    for sample_dict in sample_dicts:
                        sample_dict['image_dir'] = prompt.document.image_dir

                        # Now convert to Pydantic model
                        try:
                            sample = InstructDatasetSample(**sample_dict)
                            samples.append(sample)
                        except Exception as e:
                            logger.error(f"Pydantic validation error: {e}")
                            logger.debug(f"Sample dict: {sample_dict}")

            except OutputParserException as e:
                logger.error(f"Parse error in batch {batch_idx}: {str(e)}")
                logger.debug(f"Problematic output preview: {str(e)[:500]}")
            except Exception as e:
                logger.error(f"Unexpected error in batch {batch_idx}: {type(e).__name__}: {str(e)}")

        dataset = InstructDataset(samples=samples)
        logger.info(f"Generated {len(dataset.samples)} samples.")

        processed_datasets = cls.post_process_datasets(dataset, test_size=test_size)

        return processed_datasets

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

3. PREDICATES (must use wrappers):
   (assert (on-seg D A B))                ; D on segment AB
   (assert (on-circ P O))                 ; P on circle O
   (assert (cong (seg A B) (seg C D)))    ; AB = CD, use (seg ...) wrapper
   (assert (para (line A B) (line C D)))  ; AB // CD, use (line ...) wrapper
   (assert (perp (line A B) (line C D)))  ; AB perp CD, use (line ...) wrapper
   (assert (cong (angle A B C) (angle D E F))) ; angle equality, use (angle ...) wrapper

4. VIETNAMESE VOCABULARY:
   "nội tiếp" = incenter (inside triangle)
   "ngoại tiếp" = circumcenter (through vertices)
   "bàng tiếp" = excenter (outside triangle)

5. CRITICAL ERRORS TO AVOID:
   (cong A B C D) WRONG          ; use (assert (cong (seg A B) (seg C D)))
   (para A B C D) WRONG          ; use (assert (para (line A B) (line C D)))
   (perp l1 l2) without assert   ; use (assert (perp l1 l2))
   (intersect l1 l2 (param P point))  ; NEVER declare param inside expression
   (define O circle)                  ; If O is center, use "point" not "circle"

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

Example 8 - Segment equality (use seg wrapper):
[{
  "instruction": "Tam giác ABC, AB = AC",
  "answer": "(param (A B C) triangle)\\n(assert (cong (seg A B) (seg A C)))"
}]

Example 9 - Parallel lines (use line wrapper):
[{
  "instruction": "Tam giác ABC, BC song song với DE",
  "answer": "(param (A B C) triangle)\\n(param D point)\\n(param E point)\\n(assert (para (line B C) (line D E)))"
}]

Example 10 - Excircle (bàng tiếp = excenter):
[{
  "instruction": "Tam giác ABC, đường tròn bàng tiếp O",
  "answer": "(param (A B C) triangle)\\n(define O point (excenter A B C))"
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