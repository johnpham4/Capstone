from zenml import step
from pathlib import Path
from loguru import logger
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


@step
def merge_lora_step(
    checkpoint_path: str,
    base_model_path: str,
    output_path: str
) -> str:
    """Merge LoRA adapter back to base model"""

    logger.info(f"Loading base model from {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    logger.info(f"Loading LoRA adapter from {checkpoint_path}")
    model = PeftModel.from_pretrained(base_model, checkpoint_path)

    logger.info("Merging LoRA weights into base model")
    merged_model = model.merge_and_unload()

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving merged model to {output_dir}")
    merged_model.save_pretrained(str(output_dir))

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.save_pretrained(str(output_dir))

    logger.success(f"Merged model saved to {output_dir}")

    return str(output_dir)
