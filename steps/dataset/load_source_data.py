from typing import Annotated
from pathlib import Path
import json

from zenml import step
from loguru import logger

from llm_engineering.domains.documents import Document


@step
def load_source_data(
    source_json_path: str,
) -> Annotated[list[Document], "documents"]:
    source_path = Path(source_json_path)

    if not source_path.exists():
        raise FileNotFoundError(f"Source data not found: {source_json_path}")

    with open(source_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    for item in data:
        if "caption_vn" in item:
            doc = Document(
                caption=item.get("caption", [""])[0],
                image_dir=item.get("image", ""),
                caption_vn=item["caption_vn"]
            )
            documents.append(doc)

    logger.info(f"Loaded {len(documents)} documents from {source_json_path}")
    return documents


