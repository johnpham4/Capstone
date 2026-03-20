from typing import Annotated
from pathlib import Path
import json

from zenml import step
from loguru import logger

from pipeline.services.datasets.question_rewrite import QuestionRewriteService


@step
def load_source_data_for_question_generation(
    source_json_path: str,
) -> Annotated[list[dict], "source_items"]:
    source_path = Path(source_json_path)

    if not source_path.exists():
        raise FileNotFoundError(f"Source data not found: {source_json_path}")

    with open(source_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Source JSON must be a list of objects")

    logger.info(f"Loaded {len(data)} source items from {source_json_path}")
    return data


@step
def generate_question_dataset(
    source_items: list[dict],
    batch_size: int = 4,
    sleep_seconds: float = 2.0,
    log_every_batches: int = 10,
    max_concurrency: int = 4,
) -> Annotated[list[dict], "updated_items"]:
    indices: list[int] = []
    source_texts: list[str] = []

    for idx, item in enumerate(source_items):
        if not isinstance(item, dict):
            continue
        text = str(item.get("caption_vn", item.get("instruction", ""))).strip()
        if text:
            indices.append(idx)
            source_texts.append(text)

    logger.info(
        f"Generating problem for {len(source_texts)}/{len(source_items)} items..."
    )

    rewritten = QuestionRewriteService.rewrite_many(
        source_texts=source_texts,
        batch_size=batch_size,
        sleep_seconds=sleep_seconds,
        log_every_batches=log_every_batches,
        max_concurrency=max_concurrency,
    )

    updated_items = [dict(item) if isinstance(item, dict) else item for item in source_items]
    for idx, problem in zip(indices, rewritten):
        item = updated_items[idx]
        if isinstance(item, dict):
            item["problem"] = problem

    logger.info("Finished generating problem")
    return updated_items


@step
def save_question_dataset_to_json(
    updated_items: list[dict],
    source_json_path: str,
) -> Annotated[str, "output_path"]:
    src_path = Path(source_json_path)
    if not src_path.exists():
        raise FileNotFoundError(f"Source data not found: {source_json_path}")

    with open(src_path, "w", encoding="utf-8") as f:
        json.dump(updated_items, f, ensure_ascii=False, indent=2)

    logger.success(f"Updated source file in-place with problem: {src_path}")

    return str(src_path)
