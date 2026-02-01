from typing import Annotated, Optional
from zenml import step
from loguru import logger

from src.services.datasets.extraction import SynthGeoDatasetExtractor


@step
def download_synthgeo_dataset(
    repo_id: str = "JO-KU/SynthGeo228K",
    text_filename: str = "diagram_val.json",
    split: str = "validation",
    local_dir: Optional[str] = None,
    limit: Optional[int] = None,
) -> Annotated[list[dict], "diagram_texts"]:

    logger.info(f"Downloading SynthGeo dataset from {repo_id}")

    extractor = SynthGeoDatasetExtractor(
        repo_id=repo_id,
        local_dir=local_dir
    )

    # Download diagram text metadata
    diagram_texts = extractor.download_diagram_text(filename=text_filename)

    # Apply limit if specified
    if limit:
        diagram_texts = diagram_texts[:limit]
        logger.info(f"Limited dataset to {limit} samples")

    logger.success(f"Downloaded {len(diagram_texts)} diagram entries")

    return diagram_texts
