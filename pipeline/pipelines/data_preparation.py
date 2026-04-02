from zenml import pipeline
from loguru import logger

from pipeline.steps.data_prep import (
    download_synthgeo_dataset,
    translate_captions_to_vietnamese,
    save_prepared_dataset,
    filter_triangle
)

@pipeline
def data_preparation_pipeline(
    repo_id: str = "JO-KU/SynthGeo228K",
    text_filename: str = "diagram_val.json",
    split: str = "test",
    limit: int = None,
):
    local_dir = "./dataset/data"
    output_dir = "./dataset/data"
    output_filename = "diagrams.json"
    logger.info("Starting data preparation pipeline")

    diagram_texts = download_synthgeo_dataset(
        repo_id=repo_id,
        text_filename=text_filename,
        split=split,
        local_dir=local_dir,
        limit=limit,
    )

    diagram_texts_translated = translate_captions_to_vietnamese(
        diagram_texts=diagram_texts,
    )

    output_path = save_prepared_dataset(
        diagram_texts=diagram_texts_translated,
        output_dir=output_dir,
        output_filename=output_filename,
        image_split=split,
        repo_id=repo_id,
    )
    logger.success(f"Data preparation pipeline completed. Dataset saved to: {output_path}")

    filter_triangle(diagram_texts=diagram_texts_translated, output_dir=output_dir)

    return output_path

