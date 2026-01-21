from __future__ import annotations

from llm_engineering.domains.inference import Inference
from llm_engineering.settings import settings


class InferenceExecutor:
    def __init__(
        self,
        llm: Inference,
        query: str,
        prompt: str | None = None,
    ) -> None:
        self.llm = llm
        self.query = query

        if prompt is None:
            self.prompt = """### Instruction:
Chuyển đổi mô tả hình học tiếng Việt sang GMBL code.

GMBL Syntax chính:
- (param (A B C) triangle): Tam giác ABC thường
- (param (A B C) (iso-tri A)): Tam giác cân tại A
- (param (A B C) (right-tri B)): Tam giác vuông tại B
- (define D point (midp A B)): D là trung điểm AB
- (param D point (on-seg A B)): D nằm trên đoạn AB
- (param L line (through A)): Đường thẳng qua A
- (assert (para L1 L2)): L1 song song L2
- (assert (perp L1 L2)): L1 vuông góc L2
- (assert (on-line P L)): P nằm trên L
- (assert (= (uangle A C D) (uangle D C B))): Góc ACD = góc DCB

Ví dụ:
Input: "Tam giác ABC, AB = AC"
Output: (param (A B C) (iso-tri A))

Input: "Tam giác ABC, điểm D là trung điểm AB, điểm E là trung điểm AC"
Output: (param (A B C) triangle)
(define D point (midp A B))
(define E point (midp A C))

Bây giờ chuyển đổi:
{query}

### Response:
"""
        else:
            self.prompt = prompt

    def execute(self) -> str:
        self.llm.set_payload(
            inputs=self.prompt.format(query=self.query),
            parameters={
                "max_new_tokens": settings.MAX_NEW_TOKENS_INFERENCE,
                "repetition_penalty": 1.1,
                "temperature": settings.TEMPERATURE_INFERENCE,
                "use_cache": settings.USE_CACHE_INFERENCE,
            },
        )
        answer = self.llm.inference()[0]["generated_text"]

        return answer
