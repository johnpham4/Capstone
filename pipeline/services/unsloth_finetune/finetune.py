import argparse
import os
from pathlib import Path
from typing import Any, List, Optional

import torch
from datasets import load_dataset
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from transformers import TrainingArguments, TextStreamer
from trl import SFTTrainer, SFTConfig

from src.prompts import DSL_INFERENCE_INSTRUCTION


def load_model(
    model_name: str,
    max_seq_length: int,
    load_in_4bit: bool,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: List[str],
) -> tuple:
    # Load model with Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,  # Auto detection
        load_in_4bit=load_in_4bit,
    )

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    return model, tokenizer


def finetune(
    model_name: str = "unsloth/Qwen2.5-7B",
    output_dir: str = "/opt/ml/model",
    dataset_huggingface_workspace: str = "minn4",
    dataset_huggingface_repo_name: str = "gmbl",
    max_seq_length: int = 2048,
    load_in_4bit: bool = True,
    lora_rank: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    target_modules: List[str] = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    learning_rate: float = 2e-4,
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
) -> tuple:
    model, tokenizer = load_model(
        model_name, max_seq_length, load_in_4bit, lora_rank, lora_alpha, lora_dropout, target_modules
    )

    dataset = load_dataset(f"{dataset_huggingface_workspace}/{dataset_huggingface_repo_name}", split="train")
    if 'image' in dataset.column_names:
        dataset = dataset.remove_columns('image')

    def to_conversation(batch):
        conversations = []
        for inst, out in zip(batch["instruction"], batch["output"]):
            conversations.append([
                {
                    "role": "user",
                    "content": DSL_INFERENCE_INSTRUCTION.format(query=inst),
                },
                {
                    "role": "assistant",
                    "content": out,
                },
            ])
        return {"conversations": conversations}

    def apply_template(batch):
        texts = [
            tokenizer.apply_chat_template(
                conv,
                tokenize=False,
                add_generation_prompt=False,
            )
            for conv in batch["conversations"]
        ]
        return {"text": texts}

    dataset = dataset.map(to_conversation, batched=True)
    dataset = dataset.map(apply_template, batched=True)
    dataset = dataset.select_columns(["text"])

    sft_config = SFTConfig(
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        output_dir=output_dir,
        report_to="comet_ml",
        seed=3407,
        packing=False,
        max_seq_length=max_seq_length,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        # eval_dataset=eval_dataset,
        args=sft_config,
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    trainer.train()
    return model, tokenizer


def inference(
    model: Any,
    input_text: str = "Tam giác ABC có AB = AC",
    max_new_tokens: int = 256,
) -> None:
    model = FastLanguageModel.for_inference(model)
    message = DSL_INFERENCE_INSTRUCTION.format(query=input_text)
    inputs = tokenizer([message], return_tensors="pt").to("cuda")

    text_streamer = TextStreamer(tokenizer)
    _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=max_new_tokens, use_cache=True)


def save_model(model: Any, tokenizer: Any, output_dir: str, push_to_hub: bool = False, repo_id: Optional[str] = None):
    # Save with Unsloth's optimized method
    model.save_pretrained_merged(output_dir, tokenizer, save_method="merged_16bit")

    if push_to_hub and repo_id:
        print(f"Saving model to '{repo_id}'")
        model.push_to_hub_merged(repo_id, tokenizer, save_method="merged_16bit")


def check_if_huggingface_model_exists(model_id: str, default_value: str = "unsloth/Qwen2.5-7B") -> str:
    api = HfApi()

    try:
        api.model_info(model_id)
    except RepositoryNotFoundError:
        print(f"Model '{model_id}' does not exist.")
        model_id = default_value
        print(f"Defaulting to '{model_id}'")
        print("Train your own 'GeoUni-Qwen2.5-7B' model to avoid this behavior.")

    return model_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="unsloth/Qwen2.5-7B-Instruct")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--dataset_huggingface_workspace", type=str, default="minn4")
    parser.add_argument("--dataset_huggingface_repo_name", type=str, default="text2dsl")
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
    print(f"Training from base model '{args.model_name}'")

    output_dir_sft = Path(args.model_dir) / "output_sft"
    model, tokenizer = finetune(
        model_name=args.model_name,
        output_dir=str(output_dir_sft),
        dataset_huggingface_workspace=args.dataset_huggingface_workspace,
        dataset_huggingface_repo_name=args.dataset_huggingface_repo_name,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
    )

    inference(model, tokenizer)

    # Extract just the model name (after last /) for repo naming
    model_short_name = args.model_name.split("/")[-1]
    sft_output_model_repo_id = f"{args.model_output_huggingface_workspace}/text2diagram-{model_short_name}"
    save_model(model, tokenizer, "model_sft", push_to_hub=True, repo_id=sft_output_model_repo_id)
