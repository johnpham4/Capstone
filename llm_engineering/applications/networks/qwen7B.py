from typing import List, Optional, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_core.language_models.llms import LLM
from pydantic import PrivateAttr


class QwenLocalLLM(LLM):
    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"

    _tokenizer: Any = PrivateAttr()
    _model: Any = PrivateAttr()

    class Config:
        arbitrary_types_allowed = True

    def __init__(self):
        super().__init__()

        object.__setattr__(
            self,
            "_tokenizer",
            AutoTokenizer.from_pretrained(self.model_name),
        )

        object.__setattr__(
            self,
            "_model",
            AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                load_in_8bit=True,
            ),
        )

    @property
    def _llm_type(self) -> str:
        return "qwen-local"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
    ) -> str:
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self._model.device)

        outputs = self._model.generate(
            **inputs,
            max_new_tokens=3600,
            temperature=0.2,
        )

        return self._tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
