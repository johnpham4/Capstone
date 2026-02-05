from typing import Annotated, List, Dict
from zenml import step
from loguru import logger
from pathlib import Path
import json

from llm_src.applications.datasets.extraction import SynthGeoDatasetExtractor


@step
def filter_triangle(
    diagram_texts: List[Dict],
    output_dir: str,
    json_name: str = "diagrams_filter.json",
) -> Annotated[bool, "status"]:
    logger.info(f"Filtering {len(diagram_texts)} diagrams")

    filtered = SynthGeoDatasetExtractor.filter_diagrams(diagram_texts)

    logger.success(f"Filtered to {len(filtered)} triangle diagrams (removed {len(diagram_texts) - len(filtered)})")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_file = output_path / json_name

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    return True