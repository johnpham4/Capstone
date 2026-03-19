import argparse
import os
from pathlib import Path
from typing import Any, List, Optional

import torch
from datasets import load_dataset
from huggingface_hub import HfApi
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextStreamer,
)
from trl import SFTConfig, SFTTrainer

from src.prompts import DSL_INFERENCE_INSTRUCTION


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).lower() in {"1", "true", "yes", "y", "on"}


def _supports_bf16() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def load_model_and_tokenizer(
    model_name: str,
    load_in_4bit: bool,
    compute_dtype: torch.dtype,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: List[str],
) -> tuple:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = None
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=compute_dtype,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.config.use_cache = False
    if hasattr(model.config, "tie_word_embeddings"):
        model.config.tie_word_embeddings = False

    if load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def load_and_prepare_dataset(
    tokenizer: Any,
    dataset_huggingface_workspace: str,
    dataset_huggingface_repo_name: str,
    is_dummy: bool,
):
    dataset = load_dataset(f"{dataset_huggingface_workspace}/{dataset_huggingface_repo_name}", split="train")

    if "instruction" not in dataset.column_names or "output" not in dataset.column_names:
        raise ValueError("Dataset must contain 'instruction' and 'output' columns.")

    if "image" in dataset.column_names:
        dataset = dataset.remove_columns("image")

    if is_dummy:
        upper_bound = min(400, len(dataset))
        dataset = dataset.select(range(upper_bound))
        print(f"Dummy mode enabled. Training with {upper_bound} samples.")

    def to_prompt_completion(batch: dict) -> dict:
        prompts = []
        completions = []
        for instruction, output in zip(batch["instruction"], batch["output"], strict=False):
            if tokenizer.chat_template:
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": DSL_INFERENCE_INSTRUCTION.format(query=instruction)}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                completion = f"{output}{tokenizer.eos_token or ''}"
            else:
                prompt = f"User:\n{DSL_INFERENCE_INSTRUCTION.format(query=instruction)}\n\nAssistant:\n"
                completion = f"{output}{tokenizer.eos_token or ''}"

            prompts.append(prompt)
            completions.append(completion)

        return {"prompt": prompts, "completion": completions}

    dataset = dataset.map(to_prompt_completion, batched=True, remove_columns=dataset.column_names)
    return dataset


def finetune(
    model_name: str = "nvidia/AceMath-1.5B-Instruct",
    output_dir: str = "/opt/ml/model",
    dataset_huggingface_workspace: str = "minn4",
    dataset_huggingface_repo_name: str = "text2dsl",
    max_seq_length: int = 2048,
    load_in_4bit: bool = True,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: List[str] = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    learning_rate: float = 2e-4,
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    weight_decay: float = 0.01,
    warmup_steps: int = 10,
    train_on_responses_only: bool = True,
    is_dummy: bool = False,
) -> tuple:
    use_bf16 = _supports_bf16()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    model, tokenizer = load_model_and_tokenizer(
        model_name,
        load_in_4bit,
        compute_dtype,
        lora_rank,
        lora_alpha,
        lora_dropout,
        target_modules,
    )

    dataset = load_and_prepare_dataset(
        tokenizer=tokenizer,
        dataset_huggingface_workspace=dataset_huggingface_workspace,
        dataset_huggingface_repo_name=dataset_huggingface_repo_name,
        is_dummy=is_dummy,
    )
    print(f"Loaded {len(dataset)} formatted samples for training.")

    sft_config = SFTConfig(
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim="paged_adamw_8bit" if load_in_4bit else "adamw_torch",
        weight_decay=weight_decay,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        logging_steps=10,
        save_strategy="epoch",
        bf16=use_bf16,
        fp16=False,
        report_to="comet_ml" if os.getenv("COMET_API_KEY") else "none",
        gradient_checkpointing=True,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        output_dir=output_dir,
        seed=3407,
        max_seq_length=max_seq_length,
        packing=False,
        completion_only_loss=train_on_responses_only,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=sft_config,
    )

    if train_on_responses_only:
        print("Response-only training enabled via SFTConfig(completion_only_loss=True).")

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return model, tokenizer


def inference(
    model: Any,
    tokenizer: Any,
    input_text: str = "Tam giac ABC co AB = AC. Chung minh goc B bang goc C.",
    max_new_tokens: int = 256,
) -> None:
    prompt = DSL_INFERENCE_INSTRUCTION.format(query=input_text)

    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    streamer = TextStreamer(tokenizer)
    _ = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        streamer=streamer,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )


def save_model(
    model: Any,
    tokenizer: Any,
    output_dir: str,
    push_to_hub: bool = False,
    repo_id: Optional[str] = None,
) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    merged_model = model
    if hasattr(model, "merge_and_unload"):
        merged_model = model.merge_and_unload()

    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    if push_to_hub and repo_id:
        print(f"Saving model to '{repo_id}'")
        merged_model.push_to_hub(repo_id)
        tokenizer.push_to_hub(repo_id)


def check_if_huggingface_model_exists(model_id: str, default_value: str = "nvidia/AceMath-1.5B-Instruct") -> str:
    api = HfApi()

    try:
        api.model_info(model_id)
    except Exception:
        print(f"Model '{model_id}' does not exist or is inaccessible.")
        model_id = default_value
        print(f"Defaulting to '{model_id}'")

    return model_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="nvidia/AceMath-1.5B-Instruct")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--dataset_huggingface_workspace", type=str, default="minn4")
    parser.add_argument("--dataset_huggingface_repo_name", type=str, default="text2dsl")
    parser.add_argument("--model_output_huggingface_workspace", type=str, default="minn4")
    parser.add_argument("--load_in_4bit", type=_to_bool, default=True)
    parser.add_argument("--train_on_responses_only", type=_to_bool, default=True)
    parser.add_argument("--is_dummy", type=_to_bool, default=False)

    parser.add_argument("--output_data_dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "./outputs"))
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "./model"))
    parser.add_argument("--n_gpus", type=str, default=os.environ.get("SM_NUM_GPUS", "0"))

    args = parser.parse_args()

    print(f"Model: '{args.model_name}'")
    print(f"Num training epochs: '{args.num_train_epochs}'")
    print(f"Per device train batch size: '{args.per_device_train_batch_size}'")
    print(f"Gradient accumulation steps: '{args.gradient_accumulation_steps}'")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Warmup steps: {args.warmup_steps}")
    print(f"Dataset workspace: '{args.dataset_huggingface_workspace}'")
    print(f"Dataset repo: '{args.dataset_huggingface_repo_name}'")
    print(f"Model output workspace: '{args.model_output_huggingface_workspace}'")
    print(f"Train on responses only? '{args.train_on_responses_only}'")
    print(f"Training in dummy mode? '{args.is_dummy}'")
    print(f"Output data dir: '{args.output_data_dir}'")
    print(f"Model dir: '{args.model_dir}'")
    print(f"Number of GPUs: '{args.n_gpus}'")

    checked_model_name = check_if_huggingface_model_exists(args.model_name)
    output_dir_sft = Path(args.model_dir) / "output_sft"

    model, tokenizer = finetune(
        model_name=checked_model_name,
        output_dir=str(output_dir_sft),
        dataset_huggingface_workspace=args.dataset_huggingface_workspace,
        dataset_huggingface_repo_name=args.dataset_huggingface_repo_name,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        load_in_4bit=args.load_in_4bit,
        train_on_responses_only=args.train_on_responses_only,
        is_dummy=args.is_dummy,
    )

    inference(model=model, tokenizer=tokenizer)

    model_short_name = checked_model_name.split("/")[-1]
    model_repo_id = f"{args.model_output_huggingface_workspace}/text2diagram-{model_short_name}-peft"
    final_model_dir = str(Path(args.model_dir) / "model_sft")
    save_model(model, tokenizer, final_model_dir, push_to_hub=True, repo_id=model_repo_id)
