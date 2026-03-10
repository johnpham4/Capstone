import itertools
from typing import Annotated, Optional
from pathlib import Path
import json
from zenml import step
from loguru import logger

from src.services.datasets.extraction import SynthGeoDatasetExtractor

CHECKPOINT_FILE = Path("./dataset/data/checkpoint.json")


@step
def save_prepared_dataset(
    diagram_texts: list[dict],
    start_index: int,
    output_dir: str,
    output_filename: str,
    image_split: str,
    repo_id: str,
) -> Annotated[str, "output_path"]:

    logger.info(f"Saving prepared dataset to {output_dir}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_file = output_path / output_filename

    # Nếu file đã có, load và merge thêm vào thay vì ghi đè
    existing = []
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            existing = json.load(f)

    merged = existing + diagram_texts
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    logger.success(f"Saved dataset to {output_file} ({len(merged)} total entries)")

    logger.info(f"Downloading images from index {start_index}...")

    extractor = SynthGeoDatasetExtractor(repo_id=repo_id)
    image_dataset = extractor.load_images(split=image_split)

    # Skip stream đến start_index
    image_iter = itertools.islice(image_dataset, start_index, None)

    image_dir = output_path / "images"
    saved_count = extractor.save_images_with_captions(
        output_dir=str(image_dir),
        diagram_texts=diagram_texts,
        image_dataset=image_iter,
        limit=len(diagram_texts)
    )

    logger.success(f"Saved {saved_count} images to {image_dir}")

    # Cập nhật checkpoint sau khi cả text lẫn ảnh đã lưu xong
    new_index = start_index + len(diagram_texts)
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"last_index": new_index}, f)
    logger.info(f"Checkpoint updated: last_index = {new_index}")

    return str(output_file)
