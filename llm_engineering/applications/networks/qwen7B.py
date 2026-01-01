from typing import List, Optional, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
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
        return "qwen-local"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
    ) -> str:
        # Apply Qwen's chat template if tokenizer supports it
        if hasattr(self._tokenizer, 'apply_chat_template'):
            # Try to parse as chat messages
            try:
                # LangChain passes formatted string, we need to use it as-is
                inputs = self._tokenizer(
                    prompt,
                    return_tensors="pt",
                    add_special_tokens=True
                ).to(self._model.device)
            except Exception:
                # Fallback to direct tokenization
                inputs = self._tokenizer(
                    prompt,
                    return_tensors="pt"
                ).to(self._model.device)
        else:
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt"
            ).to(self._model.device)

        input_length = inputs["input_ids"].shape[1]

        outputs = self._model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
            top_p=0.9,
            pad_token_id=self._tokenizer.eos_token_id if self._tokenizer.eos_token_id else self._tokenizer.pad_token_id,
        )

        # Decode only the newly generated tokens (exclude the prompt)
        generated_tokens = outputs[0][input_length:]
        result = self._tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        ).strip()

        return result
