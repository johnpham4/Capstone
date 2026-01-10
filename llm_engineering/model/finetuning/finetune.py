import argparse
import os
from pathlib import Path
from typing import Any, List, Optional

import torch
from datasets import load_dataset
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    TextStreamer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from transformers import TextStreamer  # Import for streaming

gmbl_prompt = """### Instruction:
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
{}

### Response:
{}"""


def load_model(
    model_name: str,
    max_seq_length: int,
    load_in_4bit: bool,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: List[str],
) -> tuple:
    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer


def finetune(
    model_name: str = "meta-llama/Llama-2-7b-hf",
    output_dir: str = "/opt/ml/model",
    dataset_huggingface_workspace: str = "minn4",
    max_seq_length: int = 2048,
    load_in_4bit: bool = True,
    lora_rank: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    target_modules: List[str] = ["q_proj", "k_proj", "v_proj", "o_proj"],  # Llama modules
    learning_rate: float = 2e-4,
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
) -> tuple:
    model, tokenizer = load_model(
        model_name, max_seq_length, load_in_4bit, lora_rank, lora_alpha, lora_dropout, target_modules
    )
    EOS_TOKEN = tokenizer.eos_token
    print(f"Setting EOS_TOKEN to {EOS_TOKEN}")

    dataset = load_dataset(f"{dataset_huggingface_workspace}/gmbl", split="train")
    print(f"Loaded dataset with {len(dataset)} samples.")

    # Format dataset - map to text field (EXACTLY like notebook)
    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        outputs = examples["output"]
        texts = []

        for instruction, output in zip(instructions, outputs):
            text = gmbl_prompt.format(instruction, output) + EOS_TOKEN
            texts.append(text)

        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)

    print("Training dataset sample:")
    print(dataset[0]["text"][:200])

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        packing=False,
        args=TrainingArguments(
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            per_device_eval_batch_size=per_device_train_batch_size,
            warmup_steps=5,
            output_dir=output_dir,
            report_to="comet_ml",
            seed=3407,
        ),
    )

    trainer.train()

    return model, tokenizer


def inference(
    model: Any,
    tokenizer: Any,
    dataset_huggingface_workspace: str = "minn4",
    num_examples: int = 3,
) -> None:
    print("\n" + "="*80)
    print("INFERENCE - Testing model on samples")
    print("="*80)

    model.eval()

    # Load test dataset
    test_dataset = load_dataset(f"{dataset_huggingface_workspace}/gmbl", split="test")

    # Select examples: first (easy), middle, last (hard)
    indices = [0, len(test_dataset)//2, min(len(test_dataset)-1, num_examples-1)]

    from transformers import GenerationConfig
    generation_config = GenerationConfig(
        max_new_tokens=256,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    for idx in indices:
        sample = test_dataset[idx]
        instruction = sample["instruction"]
        ground_truth = sample["output"]

        print(f"\n{'='*80}")
        print(f"Example {idx + 1}:")
        print(f"{'='*80}")
        print(f"Instruction: {instruction}")
        print(f"\nGround Truth:\n{ground_truth}")
        print(f"\nModel Output:")

        # Generate
        prompt = gmbl_prompt.format(instruction, "")
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        outputs = model.generate(**inputs, generation_config=generation_config, streamer=streamer)

        print(f"{'='*80}\n")


def save_model(model: Any, tokenizer: Any, output_dir: str, push_to_hub: bool = False, repo_id: Optional[str] = None):
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    if push_to_hub and repo_id:
        print(f"Saving model to '{repo_id}'")
        model.push_to_hub(repo_id)
        tokenizer.push_to_hub(repo_id)


def check_if_huggingface_model_exists(model_id: str, default_value: str = "meta-llama/Llama-2-7b-hf") -> str:
    api = HfApi()

    try:
        api.model_info(model_id)
    except RepositoryNotFoundError:
        print(f"Model '{model_id}' does not exist.")
        model_id = default_value
        print(f"Defaulting to '{model_id}'")
        print("Train your own 'GeoUni-Llama2-7B' model to avoid this behavior.")

    return model_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--dataset_huggingface_workspace", type=str, default="minn4")
    parser.add_argument("--model_output_huggingface_workspace", type=str, default="minn4")

    parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])

    args = parser.parse_args()

    print(f"Num training epochs: '{args.num_train_epochs}'")  # noqa
    print(f"Per device train batch size: '{args.per_device_train_batch_size}'")  # noqa
    print(f"Learning rate: {args.learning_rate}")  # noqa
    print(f"Datasets will be loaded from Hugging Face workspace: '{args.dataset_huggingface_workspace}'")  # noqa
    print(f"Models will be saved to Hugging Face workspace: '{args.model_output_huggingface_workspace}'")  # noqa

    print(f"Output data dir: '{args.output_data_dir}'")  # noqa
    print(f"Model dir: '{args.model_dir}'")  # noqa
    print(f"Number of GPUs: '{args.n_gpus}'")  # noqa

    print("Starting SFT training...")  # noqa
    print(f"Training from base model '{args.model_name}'")  # noqa

    output_dir_sft = Path(args.model_dir) / "output_sft"
    model, tokenizer = finetune(
        model_name=args.model_name,
        output_dir=str(output_dir_sft),
        dataset_huggingface_workspace=args.dataset_huggingface_workspace,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
    )

    # Test inference on 3 examples from test set
    inference(model, tokenizer, dataset_huggingface_workspace=args.dataset_huggingface_workspace)

    sft_output_model_repo_id = f"{args.model_output_huggingface_workspace}/GeoUni-Llama2-7B"
    save_model(model, tokenizer, "model_sft", push_to_hub=True, repo_id=sft_output_model_repo_id)
