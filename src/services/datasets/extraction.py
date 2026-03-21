<<<<<<<< HEAD:src/services/datasets/extraction.py
import json
import re
from typing import Dict, List, Optional
from pathlib import Path

from huggingface_hub import hf_hub_download
from datasets import load_dataset
from loguru import logger


class SynthGeoDatasetExtractor:
    def __init__(
        self,
        repo_id: str = "JO-KU/SynthGeo228K",
        local_dir: Optional[str] = None
    ):
        self.repo_id = repo_id
        self.local_dir = local_dir or "./data/SynthGeo228K"


    def download_diagram_text(
        self,
        filename: str = "diagram_val.json",
    ) -> List[Dict]:

        logger.info(f"Downloading {filename}")

        path = hf_hub_download(
            repo_id=self.repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=self.local_dir,
            local_dir_use_symlinks=False
        )

        with open(path, "r", encoding="utf-8") as f:
            diagram_texts = json.load(f)

        for diagram in diagram_texts:
            if isinstance(diagram.get("caption"), list):
                diagram["caption"] = diagram["caption"][0]
                diagram["image"] = "images" + "/" + diagram["image"].split("/")[1]
            match = re.search(r"\d+", diagram["image"])
            diagram["id"] = int(match.group()) if match else None

        diagram_texts = sorted(diagram_texts, key=lambda x: x["image"])

        logger.success(f"Loaded {len(diagram_texts)} diagram texts (sorted)")

        Path.unlink(path)
        return diagram_texts

    def load_images(
        self,
        split: str = "validation",
    ):
        logger.info(f"Loading {split} split (streaming)")

        return load_dataset(
            self.repo_id,
            split=split,
            streaming=True
        )

    def save_images_with_captions(
        self,
        output_dir: str,
        diagram_texts: List[Dict],
        image_dataset,
        limit: Optional[int] = None
    ) -> int:

        output_path = Path(output_dir)
        self.img_dir = output_path
        output_path.mkdir(parents=True, exist_ok=True)

        saved = 0

        for i, (img_sample, text_sample) in enumerate(
            zip(image_dataset, diagram_texts)
        ):
            if limit is not None and i >= limit:
                break

            image = img_sample["image"]
            img_id = text_sample["id"]

            image_name = f"img_{img_id}.png"
            image.save(output_path / image_name)

            text_sample["image"] = f"images/{image_name}"
            saved += 1

        logger.success(f"Saved {saved} images")
        return saved

    def save_json(
        self,
        diagram_texts: List[Dict],
        output_file: str
    ) -> str:

        out = Path(output_file)
        self.json_path = out
        out.parent.mkdir(parents=True, exist_ok=True)

        with open(out, "w", encoding="utf-8") as f:
            json.dump(diagram_texts, f, ensure_ascii=False, indent=2)

        logger.success(f"Saved dataset JSON: {out}")
        return str(out)

    @classmethod
    def filter_diagrams(cls, diagram_texts: List[Dict]) -> List[Dict]:
        non_triangle_pattern = r"(excircle|đường tròn bàng tiếp|bàng tiếp|pentagon|ngũ giác|hexagon|lục giác|polygon|quadrilateral|tứ giác)"
        typo = r"(trianlge)"
        non_cover = r"(excenter|similar|đồng dạng|concentric|đồng tâm|inside|nằm trong|extension|kéo dài)"
        # triangle_pattern = r"\btriangle\b"

        triangle_texts = []

        for diagram in diagram_texts:
            caption = diagram["caption"]

            # Có triangle
            # if not re.search(triangle_pattern, caption, re.IGNORECASE):
            #     continue

            if re.search(non_triangle_pattern, caption, re.IGNORECASE):
                continue

            if re.search(typo, caption, re.IGNORECASE):
                continue

            if re.search(non_cover, caption, re.IGNORECASE):
                continue

            triangle_texts.append(diagram)

        return triangle_texts
========
import json
import re
from typing import Dict, List, Optional
from pathlib import Path

from huggingface_hub import hf_hub_download
from datasets import load_dataset
from loguru import logger


class SynthGeoDatasetExtractor:
    def __init__(
        self,
        repo_id: str = "JO-KU/SynthGeo228K",
        local_dir: Optional[str] = None
    ):
        self.repo_id = repo_id
        self.local_dir = local_dir or "./data/SynthGeo228K"


    def download_diagram_text(
        self,
        filename: str = "diagram_val.json",
    ) -> List[Dict]:

        logger.info(f"Downloading {filename}")

        path = hf_hub_download(
            repo_id=self.repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=self.local_dir,
            local_dir_use_symlinks=False
        )

        with open(path, "r", encoding="utf-8") as f:
            diagram_texts = json.load(f)

        for diagram in diagram_texts:
            if isinstance(diagram.get("caption"), list):
                diagram["caption"] = diagram["caption"][0]
                diagram["image"] = "images" + "/" + diagram["image"].split("/")[1]
            match = re.search(r"\d+", diagram["image"])
            diagram["id"] = int(match.group()) if match else None

        diagram_texts = sorted(diagram_texts, key=lambda x: x["image"])

        logger.success(f"Loaded {len(diagram_texts)} diagram texts (sorted)")

        Path.unlink(path)
        return diagram_texts

    def load_images(
        self,
        split: str = "validation",
    ):
        logger.info(f"Loading {split} split (streaming)")

        return load_dataset(
            self.repo_id,
            split=split,
            streaming=True
        )

    def save_images_with_captions(
        self,
        output_dir: str,
        diagram_texts: List[Dict],
        image_dataset,
        limit: Optional[int] = None
    ) -> int:

        output_path = Path(output_dir)
        self.img_dir = output_path
        output_path.mkdir(parents=True, exist_ok=True)

        saved = 0

        for i, (img_sample, text_sample) in enumerate(
            zip(image_dataset, diagram_texts)
        ):
            if limit is not None and i >= limit:
                break

            image = img_sample["image"]
            img_id = text_sample["id"]

            image_name = f"img_{img_id}.png"
            image.save(output_path / image_name)

            text_sample["image"] = f"images/{image_name}"
            saved += 1

        logger.success(f"Saved {saved} images")
        return saved

    def save_json(
        self,
        diagram_texts: List[Dict],
        output_file: str
    ) -> str:

        out = Path(output_file)
        self.json_path = out
        out.parent.mkdir(parents=True, exist_ok=True)

        with open(out, "w", encoding="utf-8") as f:
            json.dump(diagram_texts, f, ensure_ascii=False, indent=2)

        logger.success(f"Saved dataset JSON: {out}")
        return str(out)

    @classmethod
    def filter_diagrams(cls, diagram_texts: List[Dict]) -> List[Dict]:
        non_triangle_pattern = r"(excircle|quadrilateral|hexagon|polygon)"
        # triangle_pattern = r"\btriangle\b"

        triangle_texts = []

        for diagram in diagram_texts:
            caption = diagram["caption"]

            # Có triangle
            # if not re.search(triangle_pattern, caption, re.IGNORECASE):
            #     continue

            if re.search(non_triangle_pattern, caption, re.IGNORECASE):
                continue

            triangle_texts.append(diagram)

        return triangle_texts





>>>>>>>> minh-re:pipeline/services/datasets/extraction.py
