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
                        img_dir = prompt.document.image_dir
                        img_dir = "images" + img_dir.split("/")[1]
                        sample_dict['image_dir'] = img_dir

                        # Now convert to Pydantic model
                        if isinstance(sample_dict.get("answer"), list):
                            sample_dict["answer"] = "\n".join(sample_dict["answer"])
                        
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
    prompt_template_str = """Convert Vietnamese geometry problems to GMBL (Geometry Meaning-Based Language).

Bạn đang dịch ĐỀ BÀI / CAPTION HÌNH HỌC TIẾNG VIỆT sang GMBL.  
Hãy đọc kỹ từng câu tiếng Việt, xác định đúng đối tượng hình học, quan hệ và ràng buộc,  
sau đó chuyển đổi chính xác sang cú pháp GMBL bên dưới.

==============================
=== CÁC LỆNH CƠ BẢN (GMBL) ===
==============================

1. param  
Dùng để KHAI BÁO một đối tượng hình học mới (tự do), có thể kèm tham số hóa.

Cú pháp:
- (param <tên> <kiểu> <tham_số_hóa>)
- (param (<tên1> <tên2> ... <tênN>) <tham_số_hóa>)

Ví dụ tiếng Việt → GMBL:
- "Tam giác ABC" → (param (A B C) triangle)
- "Điểm D nằm trên đoạn AB" → (param D point (on-seg A B))
- "Đường thẳng d đi qua A" → (param d line (through A))

 QUAN TRỌNG:
- MỖI đối tượng chỉ được param MỘT LẦN
- Nếu một điểm / đường / đường tròn đã param → TUYỆT ĐỐI KHÔNG param lại
- Line BẮT BUỘC khai báo bằng param (KHÔNG dùng define cho line)

------------------------------------------------

2. define  
Dùng để ĐỊNH NGHĨA một đối tượng được SUY RA từ các đối tượng đã tồn tại  
(CHỈ dùng cho point / circle / number — KHÔNG dùng cho line)

Cú pháp:
- (define <tên> <kiểu> <giá_trị>)

Ví dụ:
- "D là trung điểm AB" → (define D point (midp A B))
- "O là tâm đường tròn ngoại tiếp tam giác ABC"
  → (define O point (circumcenter A B C))

------------------------------------------------

3. assert  
Dùng để THÊM RÀNG BUỘC hình học (mệnh đề đúng).

Cú pháp:
- (assert <mệnh_đề>)

Ví dụ:
- "AB song song CD"
  → (assert (para lAB lCD))
- "AB vuông góc CD"
  → (assert (perp lAB lCD))
- "AB = AC"
  → (assert (cong A B A C))

------------------------------------------------

4. eval  
Dùng để kiểm tra một mệnh đề trong mô hình cuối cùng (hiếm dùng khi dịch đề).

================
=== KIỂU DỮ LIỆU
================
- point   : điểm
- line    : đường thẳng
- circle  : đường tròn
- number  : số

====================================
=== CÁCH HIỂU TIẾNG VIỆT → GMBL ===
====================================

A. ĐIỂM – POINT

- "Điểm A" → (param A point)
- "D nằm trên đoạn AB" → (param D point (on-seg A B))
- "E nằm trên đường thẳng d" → (param E point (on-line d))
- "P nằm trên đường tròn (O)" → (param P point (on-circ C0))

------------------------------------------------

B. ĐƯỜNG THẲNG – LINE

- "Đường thẳng AB"
  → (param lAB line (through A))
     (assert (on-line B lAB))

- "Đường thẳng đi qua A và B"
  → (param l line (through A))
     (assert (on-line B l))

 LUẬT CỰC KỲ QUAN TRỌNG:
- TUYỆT ĐỐI KHÔNG dùng connecting
- Line phải được param trước khi dùng trong para / perp / on-line
- "Đường thẳng đi qua B" (chỉ 1 điểm) → KHÔNG VIẾT GMBL
- Line đã tồn tại → KHÔNG khai báo lại

------------------------------------------------

C. TAM GIÁC – TRIANGLE

- "Tam giác ABC"
  → (param (A B C) triangle)

- "Tam giác ABC vuông tại B"
  → (param (A B C) (right-tri B))

 (right-tri B) ĐÃ BAO GỒM:
- ∠ABC = 90
- TUYỆT ĐỐI KHÔNG assert lại góc này

- "Tam giác ABC cân tại A"
  → (param (A B C) (iso-tri A))

 (iso-tri A) ĐÃ BAO GỒM AB = AC  
→ KHÔNG assert (cong A B A C)

------------------------------------------------

D. GÓC – ANGLE

QUY TẮC:
- (uangle A B C) = ∠ABC (đỉnh là chữ GIỮA)

Ví dụ:
- "góc ABC = 60"
  → (assert (= (uangle A B C) 60))

- "góc BAC = 45"
  → (assert (= (uangle B A C) 45))

------------------------------------------------

E. TRUNG ĐIỂM – MIDPOINT

- "D là trung điểm AB"
  → (define D point (midp A B))

------------------------------------------------

F. ĐƯỜNG TRÒN – CIRCLE

- "Đường tròn ngoại tiếp tam giác ABC"
  → (define C0 circle (circumcircle A B C))

- "O là tâm đường tròn ngoại tiếp tam giác ABC"
  → (define O point (circumcenter A B C))

- "Đường tròn đường kính AB"
  → (define C0 circle (diam A B))

 QUAN TRỌNG:
- incircle / excircle / circumcircle → CIRCLE
- incenter / excenter / circumcenter → POINT
- Circle TUYỆT ĐỐI KHÔNG dùng trong on-seg

------------------------------------------------

G. SONG SONG – VUÔNG GÓC

- "AB song song CD"
  → (assert (para lAB lCD))

- "AB vuông góc CD"
  → (assert (perp lAB lCD))

------------------------------------------------
H. ĐƯỜNG TRÒN TIẾP XÚC TAM GIÁC

Nếu trong đề bài tiếng Việt có mô tả:
- "đường tròn nội tiếp tam giác"
- "đường tròn bàng tiếp tam giác"
- "đường tròn ngoại tiếp tam giác"
- hoặc bất kỳ câu nào nói rằng ĐƯỜNG TRÒN tiếp xúc / liên quan trực tiếp đến tam giác

THÌ BẮT BUỘC PHẢI:
1. Khai báo ĐIỂM TÂM bằng define (incenter / excenter / circumcenter)
2. ĐỒNG THỜI khai báo ĐƯỜNG TRÒN tương ứng bằng define

TUYỆT ĐỐI KHÔNG CHỈ KHAI BÁO MỖI ĐIỂM TÂM MÀ BỎ QUA ĐƯỜNG TRÒN.

Ví dụ ĐÚNG:

- "Đường tròn nội tiếp tam giác ABC, tâm I"
  → (define I point (incenter A B C))
    (define C0 circle (incircle A B C))

- "Đường tròn bàng tiếp tam giác ABC"
  → (define O point (excenter A B C))
    (define C0 circle (excircle A B C))

- "Đường tròn ngoại tiếp tam giác ABC"
  → (define O point (circumcenter A B C))
    (define C0 circle (circumcircle A B C))

QUY TẮC CỨNG:
- Nếu đề có ĐƯỜNG TRÒN → PHẢI có circle object
- Điểm tâm CHỈ LÀ PHỤ, KHÔNG ĐƯỢC THAY THẾ ĐƯỜNG TRÒN
- Không được bỏ circle trong mọi trường hợp có tiếp xúc tam giác
===========================
=== CÁC LỖI NGHIÊM CẤM ===
===========================

- KHÔNG dùng connecting
- KHÔNG param lại đối tượng đã tồn tại
- KHÔNG define line
- KHÔNG nest (foot / inter-ll / midp) trong assert
- cong CHỈ NHẬN 4 ĐIỂM
- inter-ll cần 2 LINE KHÁC NHAU
- Circle KHÔNG BAO GIỜ xuất hiện trong on-seg
- Kiểm tra cân bằng ngoặc ()


========================
=== ĐỊNH DẠNG ĐẦU RA ===
========================

CHỈ trả về JSON array:

[
  {
    "instruction": "Nguyên văn đề bài tiếng Việt",
    "answer": "Mã GMBL với \n giữa các dòng"
  }
]

QUY TẮC:
- CHỈ 2 field: instruction, answer
- KHÔNG markdown
- KHÔNG giải thích
- KHÔNG ký tự thừa


Ví dụ hoàn hảo:

1. instruction: Tam giác ABC, AB = AC, điểm D nằm trên đoạn thẳng AB, điểm E nằm trên đoạn thẳng AC, BC song song với DE, đường tròn ngoại tiếp O của tam giác ABC
answer: (param (A B C) (iso-tri A))\n(param D point (on-seg A B))\n(param E point (on-seg A C))\n(param LBC line (through B))\n(assert (on-line C LBC))\n(param LDE line (through D))\n(assert (on-line E LDE))\n(assert (para LBC LDE))\n(define O point (circumcenter A B C))\n(define C0 circle (circumcircle A B C))

2. instruction: Tam giác ABC, góc ABC = 90, góc BAC = 45, góc ACB = 45, điểm D là trung điểm của đoạn thẳng AB, điểm E là trung điểm của đoạn thẳng AC, đường thẳng DE, góc ADE = 90, đường tròn D với đường kính AB
answer: (param (A B C) (iso-tri B))\n(define D point (midp A B))\n(define E point (midp A C))\n(param LDE line (through D))\n(assert (on-line E LDE))\n(define D0 circle (diam A B))

3. instruction: Tam giác ABC, góc ABC = 90, góc BAC = 60, góc ACB = 30, điểm D nằm trên đoạn thẳng AB, điểm E nằm trên đoạn thẳng AC, BC song song với DE, đường tròn bàng tiếp O của tam giác ABC tiếp xúc với đoạn thẳng BC, góc ADE = 90
answer: (param (A B C) (right-tri B))\n(param D point (on-seg A B))\n(param E point (on-seg A C))\n(param LBC line (through B))\n(assert (on-line C LBC))\n(param LDE line (through D))\n(assert (on-line E LDE))\n(assert (para LBC LDE))\n(define O point (excenter A B C))\n(define C0 circle (excircle A B C))

...

Input: {{extract}}

Output JSON array:
"""

    @classmethod
    def post_process_datasets(cls, dataset: InstructDataset, test_size: float) -> TrainTestSplit:

        return generation_utils.create_instruct_train_test_split([dataset], test_size=test_size, random_state=42)