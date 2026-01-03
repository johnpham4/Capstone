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
    prompt_template_str = """Convert Vietnamese geometry to GMBL (Geometry Meaning-Based Language).

=== GMBL SYNTAX ===

Commands:
(param <name> <type> <parameterization>) - declare geometric object
(define <name> <type> <value>) - compute object from existing ones
(assert <predicate>) - add constraint

Types: point, line, circle

Common Functions:
midp A B - midpoint of AB
incenter/circumcenter/excenter A B C - triangle centers → point
incircle/circumcircle/excircle A B C - triangle circles → circle
foot A L1 - perpendicular foot from A to L1
inter-ll L1 L2 - intersection of two lines
connecting A B - line through A and B
perp-at A L1 - line through A perpendicular to L1
perp-bis A B - perpendicular bisector of AB
diam A B - circle with diameter AB

Predicates:
cong A B C D - |AB| = |CD| (4 points)
para L1 L2 - L1 ∥ L2 (2 lines)
perp L1 L2 - L1 ⟂ L2 (2 lines)
= N1 N2 - equality
on-seg P A B - P on segment AB (3 args)
on-circ P C - P on circle C (2 args)

Parameterizations:
triangle - (param (A B C) triangle)
(right-tri B) - right triangle at B
  CRITICAL: ALREADY contains angle ABC = 90
  NEVER add: (assert (= (uangle A B C) 90))
  NEVER add: (assert (= (uangle C B A) 90))
(iso-tri A) - isosceles with AB=AC
  CRITICAL: ALREADY contains AB = AC constraint
  NEVER add: (assert (cong A B A C))
(acute-iso-tri A) - acute isosceles at A
  CRITICAL: ALREADY contains AB = AC and acute angles
(on-seg A B) - point on segment AB
(on-circ O) - point on circle O

Shape types:
triangle, trapezoid, rectangle, square - use EXACT type from instruction
WRONG: (param (A B C D) triangle) for trapezoid
RIGHT: (param (A B C D) trapezoid)

CRITICAL:
- (uangle A B C) = angle at vertex B (middle letter)
- Parameterizations have IMPLICIT constraints - check BEFORE asserting
- Use cong A B C D for segment equality, NOT (= |A B| |C D|)

=== TRANSLATION RULES ===

Vietnamese → GMBL:
"tam giác ABC" → (param (A B C) triangle)
"tam giác ABC vuông tại B" → (param (A B C) (right-tri B))
"điểm D nằm trên AB" → (param D point (on-seg A B))
"D là trung điểm AB" → (define D point (midp A B))
"tâm nội tiếp I" → (define I point (incenter A B C))
"đường tròn nội tiếp O" → (define O circle (incircle A B C))
"đường thẳng qua A, B" → (define l line (connecting A B))
"AB song song CD" → (assert (para (connecting A B) (connecting C D)))
"AB vuông góc CD" → (assert (perp (connecting A B) (connecting C D)))
"AB = CD" → (assert (cong A B C D))
"góc ABC = 60" → (assert (= (uangle A B C) 60))
"góc ABC = góc DEF" → (assert (= (uangle A B C) (uangle D E F)))

=== CRITICAL ERRORS TO AVOID ===

1. CONFLICTING ANGLES - HIGHEST PRIORITY
   When instruction says "Tam giác ABC, góc ABC = 90" use (right-tri B)
   Then SKIP "góc ABC = 90" - DON'T assert it again!

   Example instruction: "Tam giác ABC, góc ABC = 90, góc BAC = 45"
   WRONG: (param (A B C) (right-tri B))\n(assert (= (uangle A B C) 45))
   WHY: (uangle A B C) is angle at B = 90 from right-tri, CAN'T be 45
   RIGHT: (param (A B C) (right-tri B))\n(assert (= (uangle B A C) 45))

   RULE: (right-tri B) means angle at B = 90
   - (uangle A B C) = 90 - NEVER assert this
   - (uangle C B A) = 90 - NEVER assert this
   - "góc BAC = 45" means angle at A → (uangle B A C)
   - "góc ACB = 45" means angle at C → (uangle A C B)

   CRITICAL: Read instruction angles carefully - vertex is middle letter!

2. CIRCLES in on-seg - ABSOLUTELY FORBIDDEN
   WRONG: (define O circle (excircle A B C))\n(assert (on-seg O B C))
   WRONG: (define O circle (excircle A B C))\n(assert (on-seg O (connecting B C)))
   WHY: O is CIRCLE type - NEVER EVER in on-seg
   RIGHT: (define O circle (excircle A B C)) - that's ALL, NO assertions

   "tiếp xúc với BC" = excircle IS tangent - NO on-seg needed
   Check: If variable is circle, SKIP all on-seg for that variable

3. SAME POINT in connecting - CRITICAL RULE
   WRONG: "đường thẳng đi qua điểm B" → (define l line (connecting B B))
   WHY: Need 2 DIFFERENT points for a line
   RIGHT: SKIP entire phrase - don't create line

   WRONG: (define m line (connecting B B))
   WRONG: (define l line (connecting A A))
   RIGHT: connecting needs points X Y where X ≠ Y

   If instruction says only 1 point for line, SKIP completely

4. WRONG SHAPE TYPES
   WRONG: (param (A B C D) triangle) for "hình thang"
   RIGHT: (param (A B C D) trapezoid)
   Read instruction: triangle(3), trapezoid/rectangle/square(4), pentagon(5)

5. WRONG SYNTAX for segment equality
   WRONG: (assert (= |A B| |A C|))
   RIGHT: (assert (cong A B A C))

6. WRONG VARIABLES in connecting
   WRONG: "vuông góc với AC" → (perp-at C (connecting A B))
   RIGHT: "vuông góc với AC" → (perp-at C (connecting A C))

7. NESTED FUNCTIONS in assert - NEVER DO THIS
   WRONG: (assert (on-seg (foot O (connecting B C)) B C))
   WHY: Can't use foot directly in assert
   RIGHT: (define F point (foot O (connecting B C)))\\n(assert (on-seg F B C))

   WRONG: (assert (on-seg (inter-ll L1 L2) B C))
   RIGHT: (define P point (inter-ll L1 L2))\\n(assert (on-seg P B C))

   RULE: ALWAYS define intersection/foot points in separate line FIRST
   NEVER nest foot/inter-ll/midp inside assert

   RULE: ALWAYS define intersection/foot points FIRST in separate line
   NEVER nest foot/inter-ll/midp inside assert

8. EXTRA CHARACTERS
   WRONG: ...answer ends with }]} or missing )
   RIGHT: ...answer ends with ) - balance ALL parentheses

=== OUTPUT FORMAT ===

Return ONLY JSON array with this EXACT structure:
[{
  "instruction": "Copy Vietnamese text from input",
  "answer": "GMBL code with \\n between lines"
}]

RULES:
- ONLY 2 fields: instruction and answer
- NO extra fields (variables, params, etc.)
- answer is plain text string with \\n separators
- NO markdown, NO code blocks, NO explanation

CORRECT:
[{"instruction": "Tam giác ABC vuông tại B", "answer": "(param (A B C) (right-tri B))"}]

CORRECT multi-line:
[{"instruction": "Tam giác ABC, M trung điểm BC", "answer": "(param (A B C) triangle)\\n(define M point (midp B C))"}]

WRONG: {"variables": {...}, "params": [...]}

=== VERIFICATION CHECKLIST ===

Before output, check EVERY line:
[ ] NESTED: NO foot/inter-ll inside assert - define point separately FIRST
[ ] CIRCLES: If O is circle (excircle/incircle), NEVER EVER in on-seg
[ ] "tiếp xúc BC" with excircle → define excircle only, NO on-seg
[ ] SINGLE POINT: "đường thẳng đi qua B" alone → SKIP completely
[ ] connecting: Must be (connecting X Y) where X ≠ Y, NOT (connecting B B)
[ ] ANGLES: (right-tri B) → SKIP "góc ABC = 90" from instruction
[ ] Each variable declared ONCE
[ ] All parentheses balanced
[ ] ANGLES: "góc BAC" = angle at A → (uangle B A C), middle letter is vertex
[ ] Each variable declared ONCE only
[ ] on-seg: first arg is POINT not circle
[ ] connecting: TWO DIFFERENT points (not C C)
[ ] No nested functions in assert - define first
[ ] All parentheses balanced - count ( and )
[ ] NO extra }} or }] at end

Input: {{extract}}

Output JSON array:
"""

    @classmethod
    def post_process_datasets(cls, dataset: InstructDataset, test_size: float) -> TrainTestSplit:

        return generation_utils.create_instruct_train_test_split([dataset], test_size=test_size, random_state=42)