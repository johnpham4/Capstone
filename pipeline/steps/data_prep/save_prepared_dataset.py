<<<<<<< HEAD:steps/data_prep/save_prepared_dataset.py
from typing import Annotated, Optional
from pathlib import Path
import json
from zenml import step
from loguru import logger

from src.services.datasets.extraction import SynthGeoDatasetExtractor


@step
def save_prepared_dataset(
    diagram_texts: list[dict],
    output_dir: str,
    output_filename: str,
    image_split: str,
    repo_id: str,
) -> Annotated[str, "output_path"]:

    logger.info(f"Saving prepared dataset to {output_dir}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_file = output_path / output_filename

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(diagram_texts, f, ensure_ascii=False, indent=2)

    logger.success(f"Saved dataset to {output_file}")

    logger.info("Downloading and saving images...")

    extractor = SynthGeoDatasetExtractor(repo_id=repo_id)
    image_dataset = extractor.load_images(split=image_split)

    image_dir = output_path / "images"
    saved_count = extractor.save_images_with_captions(
        output_dir=str(image_dir),
        diagram_texts=diagram_texts,
        image_dataset=image_dataset,
        limit=len(diagram_texts)
    )

    logger.success(f"Saved {saved_count} images to {image_dir}")

    return str(output_file)
=======
from typing import Annotated, Optional
from pathlib import Path
import json
from zenml import step
from loguru import logger

from pipeline.services.datasets.extraction import SynthGeoDatasetExtractor


@step
def save_prepared_dataset(
    diagram_texts: list[dict],
    output_dir: str,
    output_filename: str,
    image_split: str,
    repo_id: str,
) -> Annotated[str, "output_path"]:

    logger.info(f"Saving prepared dataset to {output_dir}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_file = output_path / output_filename

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(diagram_texts, f, ensure_ascii=False, indent=2)

    logger.success(f"Saved dataset to {output_file}")

    logger.info("Downloading and saving images...")

    extractor = SynthGeoDatasetExtractor(repo_id=repo_id)
    image_dataset = extractor.load_images(split=image_split)

    image_dir = output_path / "images"
    saved_count = extractor.save_images_with_captions(
        output_dir=str(image_dir),
        diagram_texts=diagram_texts,
        image_dataset=image_dataset,
        limit=len(diagram_texts)
    )

    logger.success(f"Saved {saved_count} images to {image_dir}")

    return str(output_file)
>>>>>>> 6cf03dda8dad8bb8fa1226b8b4e9166c3f287527:pipeline/steps/data_prep/save_prepared_dataset.py
