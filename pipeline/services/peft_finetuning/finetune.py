import argparse
from collections import Counter
from contextlib import nullcontext
import importlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from huggingface_hub import HfApi
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

try:
    from src.prompts import DSL_INFERENCE_INSTRUCTION
except ModuleNotFoundError:
    DSL_INFERENCE_INSTRUCTION: str = (
    "Chuyển bài toán hình học tiếng Việt sang Geometry DSL (S-expression).\n"
    "Chỉ trả về DSL thuần văn bản hợp lệ từ đề bài, không markdown, không giải thích.\n"
    "Bỏ qua phần yêu cầu chứng minh hoặc câu hỏi phụ, nhưng giữ mọi dữ kiện hình học và điều kiện ràng buộc trong đề.\n\n"
    "Đề bài:\n{query}\n\n"
    "DSL:"
    )


def _supports_bf16() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def _preferred_torch_dtype() -> torch.dtype:
    if _supports_bf16():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def _str2bool(value: str) -> bool:
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _format_duration(seconds: float) -> str:
    seconds = int(max(0, seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


def _sanitize_experiment_part(value: str) -> str:
    part = (value or "").strip().split("/")[-1]
    part = part.replace(" ", "").replace("_", "")
    part = re.sub(r"(?i)(-?instruct)$", "", part)
    part = re.sub(r"[^0-9A-Za-z.-]", "", part)
    return part or "unknown"


def _build_comet_experiment_name(model_name: str, dataset_repo_name: str) -> str:
    model_part = _sanitize_experiment_part(model_name)
    dataset_part = _sanitize_experiment_part(dataset_repo_name)
    return f"{model_part}_{dataset_part}"


class TrainingProgressCallback(TrainerCallback):
    """Logs training progress with elapsed time and ETA for terminal/CloudWatch visibility."""

    def __init__(self) -> None:
        self._start_time: Optional[float] = None

    def on_train_begin(self, args, state, control, **kwargs):
        self._start_time = time.time()
        print("[train-progress] Training started")
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self._start_time is None:
            return control

        max_steps = int(state.max_steps or 0)
        done_steps = int(state.global_step or 0)
        if max_steps <= 0 or done_steps <= 0:
            return control

        remaining_steps = max(max_steps - done_steps, 0)
        elapsed = time.time() - self._start_time
        speed = done_steps / elapsed if elapsed > 0 else 0.0
        eta = remaining_steps / speed if speed > 0 else 0.0
        pct = (done_steps / max_steps) * 100.0
        epoch = f"{state.epoch:.2f}" if state.epoch is not None else "n/a"

        loss = logs.get("loss") if isinstance(logs, dict) else None
        loss_text = f", loss={loss:.4f}" if isinstance(loss, (float, int)) else ""

        print(
            "[train-progress] "
            f"epoch={epoch}, "
            f"steps={done_steps}/{max_steps} ({pct:.1f}%), "
            f"remaining={remaining_steps}, "
            f"elapsed={_format_duration(elapsed)}, "
            f"eta={_format_duration(eta)}"
            f"{loss_text}"
        )
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if self._start_time is not None:
            total = time.time() - self._start_time
            print(f"[train-progress] Training finished in {_format_duration(total)}")
        return control


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
    dummy_train_samples: int,
    dummy_eval_samples: int,
):
    dataset_dict = load_dataset(f"{dataset_huggingface_workspace}/{dataset_huggingface_repo_name}")

    if "train" not in dataset_dict:
        raise ValueError("Dataset must contain a 'train' split.")

    train_dataset = dataset_dict["train"]
    eval_dataset = dataset_dict.get("validation", dataset_dict.get("val"))

    if "instruction" not in train_dataset.column_names or "answer" not in train_dataset.column_names:
        raise ValueError("Dataset must contain 'instruction' and 'answer' columns.")

    if "image" in train_dataset.column_names:
        train_dataset = train_dataset.remove_columns("image")

    if eval_dataset is not None and "image" in eval_dataset.column_names:
        eval_dataset = eval_dataset.remove_columns("image")

    if is_dummy:
        train_upper_bound = min(dummy_train_samples, len(train_dataset))
        train_dataset = train_dataset.select(range(train_upper_bound))
        if eval_dataset is not None:
            eval_upper_bound = min(dummy_eval_samples, len(eval_dataset))
            eval_dataset = eval_dataset.select(range(eval_upper_bound))
            print(
                f"Dummy mode enabled. Training with {train_upper_bound} samples, eval with {eval_upper_bound} samples."
            )
        else:
            print(f"Dummy mode enabled. Training with {train_upper_bound} samples.")

    def to_prompt_completion(batch: dict) -> dict:
        prompts = []
        completions = []
        for instruction, answer in zip(batch["instruction"], batch["answer"], strict=False):
            if tokenizer.chat_template:
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": DSL_INFERENCE_INSTRUCTION.format(query=instruction)}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                completion = f"{answer}{tokenizer.eos_token or ''}"
            else:
                prompt = f"User:\n{DSL_INFERENCE_INSTRUCTION.format(query=instruction)}\n\nAssistant:\n"
                completion = f"{answer}{tokenizer.eos_token or ''}"

            prompts.append(prompt)
            completions.append(completion)

        return {"prompt": prompts, "completion": completions}

    train_dataset = train_dataset.map(to_prompt_completion, batched=True, remove_columns=train_dataset.column_names)

    if eval_dataset is not None:
        if "instruction" not in eval_dataset.column_names or "answer" not in eval_dataset.column_names:
            raise ValueError("Validation split must contain 'instruction' and 'answer' columns.")
        eval_dataset = eval_dataset.map(to_prompt_completion, batched=True, remove_columns=eval_dataset.column_names)

    return train_dataset, eval_dataset


def _normalize_text(text: str) -> str:
    if text is None:
        return ""
    lines = [" ".join(line.strip().split()) for line in str(text).strip().splitlines() if line.strip()]
    return "\n".join(lines)


def _extract_facts(dsl: str) -> List[str]:
    normalized = _normalize_text(dsl)
    if not normalized:
        return []
    return [line for line in normalized.splitlines() if line]


def _is_valid_dsl(dsl: str) -> bool:
    text = _normalize_text(dsl)
    if not text:
        return False

    balance = 0
    for ch in text:
        if ch == "(":
            balance += 1
        elif ch == ")":
            balance -= 1
            if balance < 0:
                return False
    if balance != 0:
        return False

    for line in text.splitlines():
        if not (line.startswith("(") and line.endswith(")")):
            return False
    return True


def _fact_prf1(pred_facts: List[str], ref_facts: List[str]) -> tuple[float, float, float]:
    pred_counter = Counter(pred_facts)
    ref_counter = Counter(ref_facts)
    overlap = sum((pred_counter & ref_counter).values())
    pred_total = sum(pred_counter.values())
    ref_total = sum(ref_counter.values())
    precision = overlap / pred_total if pred_total else 0.0
    recall = overlap / ref_total if ref_total else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def _prepare_generation_context(model: Any) -> tuple[bool, torch.dtype]:
    model.eval()
    model_device = str(getattr(model, "device", "cpu"))
    use_cuda = torch.cuda.is_available() and model_device.startswith("cuda")
    if use_cuda:
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        compute_dtype = torch.float32

    if hasattr(model, "lm_head") and hasattr(model.lm_head, "to"):
        try:
            model.lm_head = model.lm_head.to(dtype=compute_dtype)
        except Exception:
            pass

    return use_cuda, compute_dtype


def _generate_dsl(
    model: Any,
    tokenizer: Any,
    instruction: str,
    max_new_tokens: int,
    use_cuda: bool,
    compute_dtype: torch.dtype,
) -> str:
    prompt_body = DSL_INFERENCE_INSTRUCTION.format(query=instruction)
    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_body}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = f"User:\n{prompt_body}\n\nAssistant:\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    autocast_context = torch.autocast(device_type="cuda", dtype=compute_dtype) if use_cuda else nullcontext()
    with torch.inference_mode():
        with autocast_context:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.08,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )

    prompt_len = inputs["input_ids"].shape[-1]
    generated = outputs[0][prompt_len:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def _resolve_split_name(dataset_dict: Any, split_alias: str) -> str:
    alias = split_alias.strip().lower()
    if alias in {"val", "valid", "validation"}:
        candidates = ["validation", "val"]
    elif alias == "test":
        candidates = ["test"]
    elif alias == "train":
        candidates = ["train"]
    else:
        candidates = [split_alias]

    for candidate in candidates:
        if candidate in dataset_dict:
            return candidate

    available = ", ".join(dataset_dict.keys())
    raise ValueError(f"Requested split '{split_alias}' not found. Available splits: {available}")


def _evaluate_single_split(
    model: Any,
    tokenizer: Any,
    dataset_split: Any,
    split_label: str,
    max_new_tokens: int,
    report_examples: int,
) -> tuple[Dict[str, float], List[Dict[str, Any]]]:
    if "instruction" not in dataset_split.column_names or "answer" not in dataset_split.column_names:
        raise ValueError(f"Split '{split_label}' must contain 'instruction' and 'answer' columns.")

    sample_count = len(dataset_split)
    em_count = 0
    valid_count = 0
    p_sum = 0.0
    r_sum = 0.0
    f1_sum = 0.0
    bad_cases: List[Dict[str, Any]] = []

    use_cuda, compute_dtype = _prepare_generation_context(model)
    start_time = time.perf_counter()
    pbar = tqdm(total=sample_count, desc=f"Evaluating {split_label}", unit="sample") if tqdm is not None else None

    for idx, sample in enumerate(dataset_split):
        pred = _generate_dsl(
            model=model,
            tokenizer=tokenizer,
            instruction=sample["instruction"],
            max_new_tokens=max_new_tokens,
            use_cuda=use_cuda,
            compute_dtype=compute_dtype,
        )
        ref = sample["answer"]

        pred_norm = _normalize_text(pred)
        ref_norm = _normalize_text(ref)

        exact_match = int(pred_norm == ref_norm)
        valid_dsl = _is_valid_dsl(pred_norm)
        precision, recall, f1 = _fact_prf1(_extract_facts(pred_norm), _extract_facts(ref_norm))

        em_count += exact_match
        valid_count += int(valid_dsl)
        p_sum += precision
        r_sum += recall
        f1_sum += f1

        if exact_match == 0 and len(bad_cases) < report_examples:
            bad_cases.append(
                {
                    "idx": idx,
                    "instruction": sample["instruction"],
                    "pred": pred_norm,
                    "ref": ref_norm,
                    "fact_f1": round(f1, 4),
                }
            )

        done = idx + 1
        remaining = sample_count - done
        elapsed = time.perf_counter() - start_time
        avg_per_sample = elapsed / done if done else 0.0
        eta = avg_per_sample * remaining

        if pbar is not None:
            pbar.set_postfix(done=done, remaining=remaining, eta_s=f"{eta:.1f}")
            pbar.update(1)
        elif done % 10 == 0 or done == sample_count:
            print(f"[{split_label}] Progress: {done}/{sample_count} | remaining={remaining} | eta~{eta:.1f}s")

    if pbar is not None:
        pbar.close()

    total_time = time.perf_counter() - start_time
    metrics = {
        "num_samples": sample_count,
        "elapsed_seconds": round(total_time, 2),
        "avg_seconds_per_sample": round(total_time / sample_count, 3) if sample_count else 0.0,
        "exact_match": round(em_count / sample_count, 4) if sample_count else 0.0,
        "valid_dsl_rate": round(valid_count / sample_count, 4) if sample_count else 0.0,
        "fact_precision_macro": round(p_sum / sample_count, 4) if sample_count else 0.0,
        "fact_recall_macro": round(r_sum / sample_count, 4) if sample_count else 0.0,
        "fact_f1_macro": round(f1_sum / sample_count, 4) if sample_count else 0.0,
    }

    print(f"Generation metrics ({split_label}):")
    for key, value in metrics.items():
        print(f"- {key}: {value}")

    return metrics, bad_cases


def evaluate_generation_metrics(
    model: Any,
    tokenizer: Any,
    dataset_huggingface_workspace: str,
    dataset_huggingface_repo_name: str,
    split_aliases: List[str],
    max_samples: Optional[int],
    max_new_tokens: int,
    report_examples: int,
) -> tuple[Dict[str, Dict[str, float]], Dict[str, List[Dict[str, Any]]]]:
    dataset_dict = load_dataset(f"{dataset_huggingface_workspace}/{dataset_huggingface_repo_name}")
    metrics_by_split: Dict[str, Dict[str, float]] = {}
    bad_cases_by_split: Dict[str, List[Dict[str, Any]]] = {}

    for split_alias in split_aliases:
        resolved_split = _resolve_split_name(dataset_dict, split_alias)
        split_dataset = dataset_dict[resolved_split]
        if max_samples is not None:
            upper_bound = min(max_samples, len(split_dataset))
            split_dataset = split_dataset.select(range(upper_bound))

        metrics, bad_cases = _evaluate_single_split(
            model=model,
            tokenizer=tokenizer,
            dataset_split=split_dataset,
            split_label=split_alias,
            max_new_tokens=max_new_tokens,
            report_examples=report_examples,
        )
        metrics_by_split[split_alias] = metrics
        bad_cases_by_split[split_alias] = bad_cases

    return metrics_by_split, bad_cases_by_split


def _parse_split_aliases(raw_splits: str) -> List[str]:
    split_aliases = [part.strip().lower() for part in str(raw_splits).split(",") if part.strip()]
    if not split_aliases:
        raise ValueError("At least one split is required for generation metrics.")
    return split_aliases


def _get_or_create_comet_experiment() -> Any:
    if not os.getenv("COMET_API_KEY"):
        return None

    try:
        comet_ml = importlib.import_module("comet_ml")
        ExistingExperiment = getattr(comet_ml, "ExistingExperiment")
        Experiment = getattr(comet_ml, "Experiment")
        get_global_experiment = getattr(comet_ml, "get_global_experiment")
    except Exception as exc:
        print(f"[comet] Unable to import comet_ml: {exc}")
        return None

    experiment = get_global_experiment()
    if experiment is not None:
        return experiment

    api_key = os.getenv("COMET_API_KEY")
    project_name = os.getenv("COMET_PROJECT_NAME")
    experiment_key = os.getenv("COMET_EXPERIMENT_KEY")

    try:
        if experiment_key:
            print(f"[comet] Attaching to existing experiment key: {experiment_key}")
            experiment = ExistingExperiment(
                api_key=api_key,
                project_name=project_name,
                previous_experiment=experiment_key,
            )
        else:
            experiment = Experiment(api_key=api_key, project_name=project_name)

        experiment_name = os.getenv("COMET_EXPERIMENT_NAME")
        if experiment_name and not experiment_key:
            experiment.set_name(experiment_name)
        return experiment
    except Exception as exc:
        print(f"[comet] Failed to initialize experiment: {exc}")
        return None


def log_generation_metrics_to_comet(metrics_by_split: Dict[str, Dict[str, float]]) -> None:
    experiment = _get_or_create_comet_experiment()
    if experiment is None:
        print("[comet] Skipping generation metric logging because no experiment is available.")
        return

    for split_name, split_metrics in metrics_by_split.items():
        payload = {
            f"{split_name}/exact_match": split_metrics.get("exact_match", 0.0),
            f"{split_name}/valid_dsl_rate": split_metrics.get("valid_dsl_rate", 0.0),
            f"{split_name}/fact_precision_macro": split_metrics.get("fact_precision_macro", 0.0),
            f"{split_name}/fact_recall_macro": split_metrics.get("fact_recall_macro", 0.0),
            f"{split_name}/fact_f1_macro": split_metrics.get("fact_f1_macro", 0.0),
            f"{split_name}/num_samples": split_metrics.get("num_samples", 0),
            f"{split_name}/avg_seconds_per_sample": split_metrics.get("avg_seconds_per_sample", 0.0),
        }
        experiment.log_metrics(payload)
        print(f"[comet] Logged generation metrics for split '{split_name}'.")


def save_generation_metrics_report(output_data_dir: str, metrics_by_split: Dict[str, Dict[str, float]]) -> Path:
    output_path = Path(output_data_dir) / "generation_metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_obj:
        json.dump(metrics_by_split, file_obj, indent=2, ensure_ascii=True)
    print(f"Saved generation metrics report to '{output_path}'")
    return output_path


def finetune(
    model_name: str = "nvidia/AceMath-1.5B-Instruct",
    output_dir: str = "/opt/ml/model",
    dataset_huggingface_workspace: str = "quangne",
    dataset_huggingface_repo_name: str = "geometry3k8-8-1-1",
    max_seq_length: int = 2048,
    load_in_4bit: bool = True,
    lora_rank: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    target_modules: List[str] = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    learning_rate: float = 1e-4,
    num_train_epochs: int = 6,
    per_device_train_batch_size: int = 4,
    per_device_eval_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.05,
    seed: int = 3407,
    enable_early_stopping: bool = True,
    early_stopping_patience: int = 2,
    early_stopping_threshold: float = 0.0,
    is_dummy: bool = False,
    dummy_train_samples: int = 400,
    dummy_eval_samples: int = 100,
) -> tuple:
    use_bf16 = _supports_bf16()
    compute_dtype = _preferred_torch_dtype()
    model, tokenizer = load_model_and_tokenizer(
        model_name,
        load_in_4bit,
        compute_dtype,
        lora_rank,
        lora_alpha,
        lora_dropout,
        target_modules,
    )

    train_dataset, eval_dataset = load_and_prepare_dataset(
        tokenizer=tokenizer,
        dataset_huggingface_workspace=dataset_huggingface_workspace,
        dataset_huggingface_repo_name=dataset_huggingface_repo_name,
        is_dummy=is_dummy,
        dummy_train_samples=dummy_train_samples,
        dummy_eval_samples=dummy_eval_samples,
    )

    print(f"Train samples: {len(train_dataset)}")
    if eval_dataset is not None:
        print(f"Eval samples: {len(eval_dataset)}")
    else:
        print("Eval split not found (expected 'validation' or 'val').")

    if os.getenv("COMET_API_KEY") and not os.getenv("COMET_EXPERIMENT_KEY"):
        comet_experiment_name = _build_comet_experiment_name(
            model_name=model_name,
            dataset_repo_name=dataset_huggingface_repo_name,
        )
        os.environ["COMET_EXPERIMENT_NAME"] = comet_experiment_name
        print(f"[comet] COMET_EXPERIMENT_NAME='{comet_experiment_name}'")

    # EarlyStoppingCallback requires evaluation + best-model tracking.
    should_eval = eval_dataset is not None
    callbacks = [TrainingProgressCallback()]
    if should_eval and enable_early_stopping:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=early_stopping_threshold,
            )
        )

    sft_config = SFTConfig(
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim="paged_adamw_8bit" if load_in_4bit else "adamw_torch",
        weight_decay=weight_decay,
        lr_scheduler_type="cosine",
        warmup_ratio=warmup_ratio,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if should_eval else "no",
        per_device_eval_batch_size=per_device_eval_batch_size,
        load_best_model_at_end=should_eval,
        metric_for_best_model="eval_loss" if should_eval else None,
        greater_is_better=False if should_eval else None,
        save_total_limit=2,
        bf16=use_bf16,
        fp16=False,
        report_to="comet_ml" if os.getenv("COMET_API_KEY") else "none",
        gradient_checkpointing=True,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        output_dir=output_dir,
        seed=seed,
        max_length=max_seq_length,
        packing=False,
        completion_only_loss=True,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=sft_config,
        callbacks=callbacks,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return model, tokenizer


def inference(
    model: Any,
    tokenizer: Any,
    input_text: str = "Xét tam giác đều WXY có đường cao WD. I thuộc XY. IL, IM vuông góc WX, WY. 1. Chứng minh WLIM nội tiếp. 2. Chứng minh IL + IM = WD.",
    max_new_tokens: int = 256,
) -> None:
    prompt_body = DSL_INFERENCE_INSTRUCTION.format(query=input_text)
    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_body}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = f"User:\n{prompt_body}\n\nAssistant:\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    model.eval()
    use_cuda = torch.cuda.is_available() and str(model.device).startswith("cuda")
    if use_cuda:
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        compute_dtype = torch.float32

    # Some PEFT/quantized runs keep lm_head in float32 while hidden states are bf16/fp16.
    if hasattr(model, "lm_head") and hasattr(model.lm_head, "to"):
        try:
            model.lm_head = model.lm_head.to(dtype=compute_dtype)
        except Exception:
            pass

    with torch.inference_mode():
        if use_cuda:
            with torch.autocast(device_type="cuda", dtype=compute_dtype):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    repetition_penalty=1.08,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
        else:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.08,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

    prompt_len = inputs["input_ids"].shape[-1]
    generated = outputs[0][prompt_len:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    print(text)


def save_adapter_model(
    model: Any,
    tokenizer: Any,
    output_dir: str,
) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def merge_adapter_to_full_precision_base(
    base_model_name: str,
    adapter_path_or_repo: str,
    device_map: str = "auto",
) -> Any:
    merge_dtype = _preferred_torch_dtype()
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        torch_dtype=merge_dtype,
        device_map=device_map,
    )
    peft_model = PeftModel.from_pretrained(base_model, adapter_path_or_repo)
    return peft_model.merge_and_unload()


def load_hf_model_for_inference(model_id: str) -> tuple:
    infer_dtype = _preferred_torch_dtype()
    loaded_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=infer_dtype,
        device_map="auto",
    )
    loaded_tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
    if loaded_tokenizer.pad_token is None:
        loaded_tokenizer.pad_token = loaded_tokenizer.eos_token
    return loaded_model, loaded_tokenizer


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
    parser.add_argument("--num_train_epochs", type=int, default=6)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--enable_early_stopping", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--early_stopping_patience", type=int, default=2)
    parser.add_argument("--early_stopping_threshold", type=float, default=0.0)
    parser.add_argument("--dataset_huggingface_workspace", type=str, default="quangne")
    parser.add_argument("--dataset_huggingface_repo_name", type=str, default="geometry3k8-8-1-1")
    parser.add_argument("--model_output_huggingface_workspace", type=str, default="quangne")
    parser.add_argument("--merged_repo_id", type=str, default="")
    parser.add_argument("--verify_from_hub", type=_str2bool, default=False)
    parser.add_argument(
        "--verify_input_text",
        type=str,
        default="Cho tam giác ABC, M là trung điểm của BC. Trên tia đối của BA lấy điểm N sao cho BN = AB. Gọi I là giao điểm MN và AC. Chứng minh AI = 2IC",
    )
    parser.add_argument("--is_dummy", type=_str2bool, default=False)
    parser.add_argument("--dummy_train_samples", type=int, default=400)
    parser.add_argument("--dummy_eval_samples", type=int, default=100)
    parser.add_argument("--evaluate_only", type=_str2bool, default=False)
    parser.add_argument(
        "--evaluation_model_id",
        type=str,
        default="",
        help="Model id/path used when --evaluate_only=true (for example a merged model on HF Hub).",
    )
    parser.add_argument(
        "--compute_generation_metrics",
        type=_str2bool,
        default=True,
        help="Compute generation metrics (exact_match, valid_dsl_rate, fact_precision_macro, fact_recall_macro).",
    )
    parser.add_argument(
        "--generation_metrics_splits",
        type=str,
        default="val,test",
        help="Comma-separated split aliases for generation metrics, for example: val,test",
    )
    parser.add_argument(
        "--generation_metrics_max_samples",
        type=int,
        default=0,
        help="Limit generation metric evaluation size per split. 0 means full split.",
    )
    parser.add_argument(
        "--generation_metrics_max_new_tokens",
        type=int,
        default=256,
        help="Max new tokens for generation during metric evaluation.",
    )
    parser.add_argument(
        "--generation_metrics_report_examples",
        type=int,
        default=5,
        help="Max mismatched examples kept in memory per split.",
    )
    parser.add_argument(
        "--comet_experiment_key",
        type=str,
        default="",
        help="Existing Comet experiment key to append metrics to instead of creating a new experiment.",
    )

    parser.add_argument("--output_data_dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "./outputs"))
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "./model"))
    parser.add_argument("--n_gpus", type=str, default=os.environ.get("SM_NUM_GPUS", "0"))

    args = parser.parse_args()

    print(f"Model: '{args.model_name}'")
    print(f"Num training epochs: '{args.num_train_epochs}'")
    print(f"Per device train batch size: '{args.per_device_train_batch_size}'")
    print(f"Per device eval batch size: '{args.per_device_eval_batch_size}'")
    print(f"Gradient accumulation steps: '{args.gradient_accumulation_steps}'")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Warmup ratio: {args.warmup_ratio}")
    print(f"Seed: {args.seed}")
    print(f"Early stopping enabled: {args.enable_early_stopping}")
    print(f"Early stopping patience: {args.early_stopping_patience}")
    print(f"Early stopping threshold: {args.early_stopping_threshold}")
    print(f"Dataset workspace: '{args.dataset_huggingface_workspace}'")
    print(f"Dataset repo: '{args.dataset_huggingface_repo_name}'")
    print(f"Model output workspace: '{args.model_output_huggingface_workspace}'")
    print(f"Merged repo override: '{args.merged_repo_id}'")
    print(f"Verify from hub after push: {args.verify_from_hub}")
    print(f"Evaluate only mode: {args.evaluate_only}")
    print(f"Evaluation model id: '{args.evaluation_model_id}'")
    print(f"Compute generation metrics: {args.compute_generation_metrics}")
    print(f"Generation metric splits: '{args.generation_metrics_splits}'")
    print(f"Generation metric max samples: {args.generation_metrics_max_samples}")
    print(f"Generation metric max new tokens: {args.generation_metrics_max_new_tokens}")
    print(f"Generation metric report examples: {args.generation_metrics_report_examples}")
    print(f"Comet experiment key override: '{args.comet_experiment_key}'")
    print(f"Dummy train samples: {args.dummy_train_samples}")
    print(f"Dummy eval samples: {args.dummy_eval_samples}")
    print(f"Output data dir: '{args.output_data_dir}'")
    print(f"Model dir: '{args.model_dir}'")
    print(f"Number of GPUs: '{args.n_gpus}'")

    if args.comet_experiment_key:
        os.environ["COMET_EXPERIMENT_KEY"] = args.comet_experiment_key
        print(f"[comet] Using existing experiment key: '{args.comet_experiment_key}'")

    split_aliases = _parse_split_aliases(args.generation_metrics_splits)
    max_metric_samples = args.generation_metrics_max_samples if args.generation_metrics_max_samples > 0 else None

    if os.getenv("COMET_API_KEY") and not os.getenv("COMET_EXPERIMENT_KEY"):
        comet_experiment_name = _build_comet_experiment_name(
            model_name=args.evaluation_model_id or args.model_name,
            dataset_repo_name=args.dataset_huggingface_repo_name,
        )
        os.environ["COMET_EXPERIMENT_NAME"] = comet_experiment_name
        print(f"[comet] COMET_EXPERIMENT_NAME='{comet_experiment_name}'")

    if args.evaluate_only:
        if not args.evaluation_model_id:
            raise ValueError("--evaluation_model_id is required when --evaluate_only=true")

        print(f"Loading model for evaluate-only mode: '{args.evaluation_model_id}'")
        model, tokenizer = load_hf_model_for_inference(args.evaluation_model_id)

        if args.compute_generation_metrics:
            metrics_by_split, _ = evaluate_generation_metrics(
                model=model,
                tokenizer=tokenizer,
                dataset_huggingface_workspace=args.dataset_huggingface_workspace,
                dataset_huggingface_repo_name=args.dataset_huggingface_repo_name,
                split_aliases=split_aliases,
                max_samples=max_metric_samples,
                max_new_tokens=args.generation_metrics_max_new_tokens,
                report_examples=args.generation_metrics_report_examples,
            )
            log_generation_metrics_to_comet(metrics_by_split)
            save_generation_metrics_report(args.output_data_dir, metrics_by_split)

        inference(
            model=model,
            tokenizer=tokenizer,
            input_text=args.verify_input_text,
            max_new_tokens=args.generation_metrics_max_new_tokens,
        )
        raise SystemExit(0)

    checked_model_name = check_if_huggingface_model_exists(args.model_name)
    output_dir_sft = Path(args.model_dir) / "output_sft"

    model, tokenizer = finetune(
        model_name=checked_model_name,
        output_dir=str(output_dir_sft),
        dataset_huggingface_workspace=args.dataset_huggingface_workspace,
        dataset_huggingface_repo_name=args.dataset_huggingface_repo_name,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        seed=args.seed,
        enable_early_stopping=args.enable_early_stopping,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=args.early_stopping_threshold,
        is_dummy=args.is_dummy,
        dummy_train_samples=args.dummy_train_samples,
        dummy_eval_samples=args.dummy_eval_samples,
    )

    if args.compute_generation_metrics:
        metrics_by_split, _ = evaluate_generation_metrics(
            model=model,
            tokenizer=tokenizer,
            dataset_huggingface_workspace=args.dataset_huggingface_workspace,
            dataset_huggingface_repo_name=args.dataset_huggingface_repo_name,
            split_aliases=split_aliases,
            max_samples=max_metric_samples,
            max_new_tokens=args.generation_metrics_max_new_tokens,
            report_examples=args.generation_metrics_report_examples,
        )
        log_generation_metrics_to_comet(metrics_by_split)
        save_generation_metrics_report(args.output_data_dir, metrics_by_split)

    inference(model=model, tokenizer=tokenizer)

    model_short_name = checked_model_name.split("/")[-1]
    merged_dataset_tag = _sanitize_experiment_part(args.dataset_huggingface_repo_name)
    merged_repo_id = (
        args.merged_repo_id
        if args.merged_repo_id
        else f"{args.model_output_huggingface_workspace}/text2diagram-{model_short_name}-merged-{merged_dataset_tag}"
    )

    adapter_output_dir = str(Path(args.model_dir) / "adapter_sft")
    merged_output_dir = str(Path(args.model_dir) / "model_sft_merged")

    # Keep adapter local-only for merge; only merged model is pushed to Hub.
    save_adapter_model(model, tokenizer, adapter_output_dir)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    merged_model = merge_adapter_to_full_precision_base(
        base_model_name=checked_model_name,
        adapter_path_or_repo=adapter_output_dir,
    )
    save_model(merged_model, tokenizer, merged_output_dir, push_to_hub=True, repo_id=merged_repo_id)

    if args.verify_from_hub:
        print(f"Loading merged model back from Hub for verification: '{merged_repo_id}'")
        loaded_model, loaded_tokenizer = load_hf_model_for_inference(merged_repo_id)
        inference(
            model=loaded_model,
            tokenizer=loaded_tokenizer,
            input_text=args.verify_input_text,
        )
