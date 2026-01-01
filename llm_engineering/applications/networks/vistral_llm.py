from typing import List, Optional, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain_core.language_models.llms import LLM
from pydantic import PrivateAttr


class VistralLocalLLM(LLM):
    """Vistral-7B-Chat model optimized for Vietnamese language tasks."""
    model_name: str = "Viet-Mistral/Vistral-7B-Chat"

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

        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )

        object.__setattr__(
            self,
            "_model",
            AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                quantization_config=quantization_config,
            ),
        )

    @property
    def _llm_type(self) -> str:
        return "vistral-local"

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
            temperature=0.1,  # Lower temperature for more deterministic output
            do_sample=True,
            top_p=0.95,
        )

        return self._tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
