import re
import json
from typing import Dict, List, Optional
from pathlib import Path
from huggingface_hub import hf_hub_download
from datasets import load_dataset
from loguru import logger


class SynthGeoDatasetExtractor:
    """Service for extracting and processing SynthGeo228K dataset"""

    def __init__(
        self,
        repo_id: str = "JO-KU/SynthGeo228K",
        local_dir: Optional[str] = None
    ):
        self.repo_id = repo_id
        self.local_dir = local_dir or "./data/SynthGeo228K"

    @staticmethod
    def get_diagram_id(text: str) -> int:
        match = re.search(r"\d+", text)
        if not match:
            raise ValueError(f"Invalid diagram text: {text}")
        return int(match.group())

    def download_diagram_text(
        self,
        filename: str = "diagram_val.json",
    ) -> List[Dict]:

        logger.info(f"Downloading {filename} from {self.repo_id}")

        path = hf_hub_download(
            repo_id=self.repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=self.local_dir,
            local_dir_use_symlinks=False
        )

        with open(path, "r", encoding="utf-8") as f:
            diagram_texts = json.load(f)

            # Process each diagram entry
            for diagram in diagram_texts:
                diagram["id"] = self.get_diagram_id(diagram["image"])

                # Handle caption format - could be list or string
                if isinstance(diagram.get("caption"), list):
                    diagram["caption"] = diagram["caption"][0]

            # Sort by ID for consistent ordering
            diagram_texts = sorted(diagram_texts, key=lambda x: x["id"])

        # Delete the downloaded file to save space
        Path(path).unlink(missing_ok=True)
        logger.info(f"Deleted temporary file: {path}")

        logger.info(f"Loaded {len(diagram_texts)} diagram texts")
        return diagram_texts

    def load_images(
        self,
        split: str = "test",
        streaming: bool = True
    ):
        logger.info(f"Loading {split} split from {self.repo_id}")

        ds = load_dataset(
            self.repo_id,
            split=split,
            streaming=streaming
        )

        return ds

    def save_images_with_captions(
        self,
        output_dir: str,
        diagram_texts: List[Dict],
        image_dataset,
        limit: Optional[int] = None
    ) -> int:

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        saved_count = 0

        for i, (img_sample, text_sample) in enumerate(zip(image_dataset, diagram_texts)):
            if limit is not None and i >= limit:
                break

            image = img_sample["image"]
            diagram_id = text_sample["id"]

            # Save only the image file (no separate JSON)
            filename = output_path / f"img_{diagram_id}.png"
            image.save(filename)

            # Update diagram_texts with relative image path
            text_sample["image"] = f"images/img_{diagram_id}.png"

            saved_count += 1

        logger.success(f"Saved {saved_count} images to {output_path}")
        return saved_count

    def create_combined_dataset(
        self,
        diagram_texts: List[Dict],
        output_file: str
    ) -> str:

        output_path = Path(output_file)
        output_path.parent.mkdir(exist_ok=True, parents=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(diagram_texts, f, ensure_ascii=False, indent=2)

        logger.success(f"Saved combined dataset to {output_path}")
        return str(output_path)

    def extract_and_process(
        self,
        text_filename: str = "diagram_val.json",
        split: str = "test",
        output_dir: Optional[str] = None,
        save_images: bool = False,
        limit: Optional[int] = None
    ) -> tuple[List[Dict], str]:

        # Download and process text metadata
        diagram_texts = self.download_diagram_text(filename=text_filename)

        # Load images
        image_dataset = self.load_images(split=split, streaming=True)

        # Apply limit if specified
        if limit:
            diagram_texts = diagram_texts[:limit]

        # Save images if requested
        if save_images:
            img_output_dir = output_dir or f"{self.local_dir}/images"
            self.save_images_with_captions(
                output_dir=img_output_dir,
                diagram_texts=diagram_texts,
                image_dataset=image_dataset,
                limit=limit
            )

        # Create combined dataset file
        dataset_output = output_dir or self.local_dir
        output_file = f"{dataset_output}/processed_dataset.json"
        self.create_combined_dataset(diagram_texts, output_file)

        return diagram_texts, output_file
