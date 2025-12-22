from pydantic import BaseModel


class TrainingConfig(BaseModel):
    vq_model_path: str
    base_llm_path: str
    output_dir: str
    batch_size: int = 1
    epochs: int = 3
    learning_rate: float = 2e-4
    lora_r: int = 32
    lora_alpha: int = 64
    gradient_accumulation_steps: int = 8
    use_8bit: bool = True
    gradient_checkpointing: bool = True
    fp16: bool = True
