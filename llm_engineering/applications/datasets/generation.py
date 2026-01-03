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
    prompt_template_str = """Convert Vietnamese geometry description to GMBL (Geometry Meaning-Based Language).

=== SYNTAX ===

1. PARAM - Declare base objects:
   (param A point)
   (param (A B C) triangle)
   (param (A B C) (right-tri B))
   (param D point (on-seg A B))

2. DEFINE - Compute derived objects:
   (define M point (midp A B))
   (define O point (incenter A B C))
   (define l line (connecting A B))
   (define c circle (incircle A B C))

3. ASSERT - State constraints:
   (assert (cong A B C D))
   (assert (para L1 L2))
   (assert (perp L1 L2))
   (assert (= (uangle A B C) (uangle D E F)))

=== TYPE RULES ===

Centers are POINTS:
- (incenter A B C) returns point
- (circumcenter A B C) returns point
- (excenter A B C) returns point

Circles are CIRCLES:
- (incircle A B C) returns circle
- (circumcircle A B C) returns circle
- (excircle A B C) returns circle

=== ANGLE NOTATION ===

(uangle A B C) - angle at vertex B (middle letter)
- góc ABC (at vertex B) → (uangle A B C)
- góc BAC (at vertex A) → (uangle B A C)
- góc ACB (at vertex C) → (uangle A C B)

CRITICAL: Middle letter is the vertex
- (uangle A B C) ≠ (uangle B A C) ≠ (uangle A C B)
- Each angle must use correct vertex position

=== KEY FUNCTIONS ===

Points:
- (midp A B) - midpoint
- (incenter A B C) - incenter
- (circumcenter A B C) - circumcenter
- (excenter A B C) - excenter
- (orthocenter A B C) - orthocenter
- (centroid A B C) - centroid
- (foot A L1) - perpendicular foot
- (inter-ll L1 L2) - line intersection

Lines:
- (connecting A B) - line through points
- (perp-bis A B) - perpendicular bisector
- (perp-at A L1) - perpendicular at point

Circles:
- (incircle A B C) - incircle
- (circumcircle A B C) - circumcircle
- (excircle A B C) - excircle
- (diam A B) - circle with diameter

Predicates:
- (cong A B C D) - segment equality
- (para L1 L2) - parallel
- (perp L1 L2) - perpendicular
- (= (uangle A B C) N) - angle equality
- (on-circ P C) - point on circle (NOT on-seg)
- (on-seg P A B) - point on segment

=== CRITICAL CONSTRAINTS ===

1. All variables must be declared before use
   - Check param and define statements
   - Never use undefined variables in assert

2. Angle vertices must match
   - góc ABC → (uangle A B C) at vertex B
   - góc BAC → (uangle B A C) at vertex A
   - góc ACB → (uangle A C B) at vertex C

3. Point on circle requires on-circ
   - "điểm P nằm trên đường tròn O" → (param P point (on-circ O))
   - NOT on-seg when O is circle

4. Line requires two different points
   - (connecting A B) valid only if A ≠ B
   - NEVER use (connecting A A) - invalid
   - If instruction mentions line without two points, infer from context
   - If no context available, use perp-at or other constructors

5. No duplicate variable declarations
   - Use param OR define, never both for same variable

6. Handle incomplete instructions carefully
   - "đường thẳng qua C vuông góc AB" → use (perp-at C (connecting A B))
   - "đường thẳng qua C" alone → skip if no second point available
   - Never hallucinate undefined points

=== VIETNAMESE MAPPING ===

"tam giác ABC" → (param (A B C) triangle)
"tam giác ABC vuông tại B" → (param (A B C) (right-tri B))
"tam giác ABC cân tại A" → (param (A B C) (iso-tri A))
"điểm M nằm trên BC" → (param M point (on-seg B C))
"M là trung điểm BC" → (define M point (midp B C))
"tâm nội tiếp I" → (define I point (incenter A B C))
"đường tròn nội tiếp O" → (define O circle (incircle A B C))
"đường tròn bàng tiếp E" → (define E circle (excircle A B C))
"đường thẳng qua A và B" → (define l line (connecting A B))
"AB song song CD" → (assert (para (connecting A B) (connecting C D)))
"AB vuông góc CD" → (assert (perp (connecting A B) (connecting C D)))
"AB = CD" → (assert (cong A B C D))
"góc ABC = góc DEF" → (assert (= (uangle A B C) (uangle D E F)))

=== EXAMPLES ===

Input: "Tam giác ABC"
Output:
[{
  "instruction": "Tam giác ABC",
  "answer": "(param (A B C) triangle)"
}]

Input: "Tam giác ABC vuông tại B"
Output:
[{
  "instruction": "Tam giác ABC vuông tại B",
  "answer": "(param (A B C) (right-tri B))"
}]

Input: "Tam giác ABC, điểm M là trung điểm BC"
Output:
[{
  "instruction": "Tam giác ABC, điểm M là trung điểm BC",
  "answer": "(param (A B C) triangle)\\n(define M point (midp B C))"
}]

Input: "Tam giác ABC, đường tròn nội tiếp O"
Output:
[{
  "instruction": "Tam giác ABC, đường tròn nội tiếp O",
  "answer": "(param (A B C) triangle)\\n(define O circle (incircle A B C))"
}]

Input: "Tam giác ABC, đường thẳng đi qua B và C"
Output:
[{
  "instruction": "Tam giác ABC, đường thẳng đi qua B và C",
  "answer": "(param (A B C) triangle)\\n(define l line (connecting B C))"
}]

Input: "Tam giác ABC, AB = AC"
Output:
[{
  "instruction": "Tam giác ABC, AB = AC",
  "answer": "(param (A B C) triangle)\\n(assert (cong A B A C))"
}]

Input: "Tam giác ABC, BC song song DE"
Output:
[{
  "instruction": "Tam giác ABC, BC song song DE",
  "answer": "(param (A B C) triangle)\\n(param D point)\\n(param E point)\\n(define l1 line (connecting B C))\\n(define l2 line (connecting D E))\\n(assert (para l1 l2))"
}]

Input: "Tam giác ABC, đường thẳng đi qua B và C"
Output:
[{
  "instruction": "Tam giác ABC, đường thẳng đi qua B và C",
  "answer": "(param (A B C) triangle)\\n(define l line (connecting B C))"
}]

Input: "Tam giác ABC, đường thẳng qua C vuông góc AB"
Output:
[{
  "instruction": "Tam giác ABC, đường thẳng qua C vuông góc AB",
  "answer": "(param (A B C) triangle)\\n(define l line (perp-at C (connecting A B)))"
}]

=== COMMON ERRORS ===

Error: Using undefined variables
Wrong: (assert (= (uangle A B C) (uangle D E F)))  [D, E, F not declared]
Right: (assert (= (uangle A B C) 90))

Error: Wrong angle vertex
Wrong: góc BAC = 45 → (assert (= (uangle A B C) 45))  [vertex is A not B]
Right: góc BAC = 45 → (assert (= (uangle B A C) 45))

Error: Wrong parameterization for circle
Wrong: điểm N trên đường tròn O → (param N point (on-seg O))
Right: điểm N trên đường tròn O → (param N point (on-circ O))

Error: Confusing center point vs circle
Wrong: (define O point (incenter A B C))  [for "đường tròn nội tiếp O"]
Right: (define O circle (incircle A B C))

Error: Same point in connecting
Wrong: đường thẳng qua C → (define l line (connecting C C))
Right: Use perp-at or skip if no second point

Error: Hallucinating undefined points
Wrong: đường thẳng qua C → (define l line (connecting C E))  [E not declared]
Right: Only use declared variables or infer from triangle vertices

Error: Double declaration
Wrong: (param O circle)\n(define O circle (diam A B))
Right: (define O circle (diam A B))

=== REQUIREMENTS ===

1. Declare all variables before use
2. Use correct types: point vs circle
3. Use cong for segments, = for angles
4. Never declare same variable twice
5. Create all mentioned objects
6. Balance parentheses
7. Use \\n for line breaks
8. Return JSON array with one object
9. No markdown, no explanation

=== TASK ===

Input: {{extract}}

Output (JSON only):
"""

    @classmethod
    def post_process_datasets(cls, dataset: InstructDataset, test_size: float) -> TrainTestSplit:

        return generation_utils.create_instruct_train_test_split([dataset], test_size=test_size, random_state=42)