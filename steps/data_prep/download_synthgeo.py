import json
from pathlib import Path
from typing import Annotated, Optional
from zenml import step
from loguru import logger

from src.services.datasets.extraction import SynthGeoDatasetExtractor

CHECKPOINT_FILE = Path("./dataset/data/checkpoint.json")

@step
def download_synthgeo_dataset(
    repo_id: str = "JO-KU/SynthGeo228K",
    text_filename: str = "diagram_val.json",
    split: str = "validation",
    local_dir: Optional[str] = None,
    limit: Optional[int] = None,
) -> tuple[
    Annotated[list[dict], "diagram_texts"],
    Annotated[int, "start_index"]
    ]:

    logger.info(f"Downloading SynthGeo dataset from {repo_id}")

    # Read checkpoint
    start_index = 0
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            start_index = json.load(f).get("last_index", 0)
        logger.info(f"Resuming from text checkpoint: index {start_index}")


    extractor = SynthGeoDatasetExtractor(
        repo_id=repo_id,
        local_dir=local_dir
    )

    # Download diagram text metadata
    diagram_texts = extractor.download_diagram_text(filename=text_filename)

    # Apply limit if specified
    if limit:
        diagram_texts = diagram_texts[start_index : start_index + limit]
        logger.info(f"Limited dataset to {limit} samples")
    else:
        diagram_texts = diagram_texts[start_index:]
        

    logger.success(f"Downloaded {len(diagram_texts)} diagram entries")

    return diagram_texts, start_index
