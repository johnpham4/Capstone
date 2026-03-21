"""
Split diagrams.json and corresponding images into N equal folders.

Usage:
    python tools/split_dataset.py
    python tools/split_dataset.py --input-json dataset/data/diagrams.json --output-dir dataset/data/splits --num-splits 4
"""

import json
import shutil
import click
from pathlib import Path
from loguru import logger


@click.command()
@click.option(
    "--input-json",
    default="dataset/data/diagrams_filter.json",
    help="Path to the input diagrams JSON file",
)
@click.option(
    "--images-dir",
    default="dataset/data/images",
    help="Path to the folder containing images",
)
@click.option(
    "--output-dir",
    default="dataset/data/splits",
    help="Parent output directory for the split folders",
)
@click.option(
    "--names",
    default="Minh,Nhi,Quang,Khang",
    help="Comma-separated folder names for each split",
)
@click.option(
    "--output-json-name",
    default="diagrams.json",
    help="Name of the JSON file in each split folder",
)
def split_dataset(
    input_json: str,
    images_dir: str,
    output_dir: str,
    names: str,
    output_json_name: str,
):
    input_path = Path(input_json)
    images_path = Path(images_dir)
    output_path = Path(output_dir)

    assert input_path.exists(), f"Input JSON not found: {input_path}"
    assert images_path.exists(), f"Images directory not found: {images_path}"

    with open(input_path, "r", encoding="utf-8") as f:
        diagrams = json.load(f)

    split_names = [n.strip() for n in names.split(",")]
    num_splits = len(split_names)
    total = len(diagrams)
    chunk_size = total // num_splits
    remainder = total % num_splits

    logger.info(f"Total entries: {total}")
    logger.info(f"Splitting into {num_splits} folders (~{chunk_size} each): {split_names}")

    output_path.mkdir(parents=True, exist_ok=True)

    start = 0
    for i, name in enumerate(split_names):
        # Distribute remainder entries across first splits
        end = start + chunk_size + (1 if i < remainder else 0)
        chunk = diagrams[start:end]

        split_dir = output_path / name
        split_images_dir = split_dir / "images"
        split_images_dir.mkdir(parents=True, exist_ok=True)

        # Update image paths to point to local images/ subfolder and copy images
        chunk_updated = []
        missing = 0
        for entry in chunk:
            img_filename = Path(entry["image"]).name  # e.g. img_1.png
            src = images_path / img_filename
            dst = split_images_dir / img_filename

            if src.exists():
                shutil.copy2(src, dst)
            else:
                missing += 1
                logger.warning(f"Image not found, skipping copy: {src}")

            entry_copy = dict(entry)
            entry_copy["image"] = f"images/{img_filename}"
            chunk_updated.append(entry_copy)

        # Save JSON
        json_path = split_dir / output_json_name
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(chunk_updated, f, ensure_ascii=False, indent=2)

        logger.success(
            f"{name}: {len(chunk_updated)} entries, "
            f"{len(chunk_updated) - missing} images copied"
            + (f", {missing} missing" if missing else "")
        )

        start = end

    # Xóa file và folder gốc sau khi split xong
    shutil.rmtree(images_path)
    logger.info(f"Deleted original images directory: {images_path}")

    input_path.unlink()
    logger.info(f"Deleted original JSON file: {input_path}")

    logger.success(f"Done. Output: {output_path}")


if __name__ == "__main__":
    split_dataset()


