from typing import Annotated
from zenml import step
from loguru import logger
from tqdm.auto import tqdm

from pipeline.services.preprocessing.translation import GeometryTranslator


@step
def translate_captions_to_vietnamese(
    diagram_texts: list[dict],
) -> Annotated[list[dict], "diagram_texts_translated"]:

    logger.info(f"Translating {len(diagram_texts)} captions to Vietnamese")

    translator = GeometryTranslator()

    # Translate each caption
    for diagram in tqdm(diagram_texts, desc="Translating captions"):
        english_caption = diagram.get("caption", "")

        if english_caption:
            vietnamese_caption = translator.translate(english_caption)
            diagram["caption_vn"] = vietnamese_caption
        else:
            logger.warning(f"Empty caption for diagram ID {diagram.get('id', 'unknown')}")
            diagram["caption_vn"] = ""

    logger.success(f"Translated {len(diagram_texts)} captions")

    return diagram_texts
