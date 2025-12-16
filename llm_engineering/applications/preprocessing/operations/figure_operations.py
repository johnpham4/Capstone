from pathlib import Path
from typing import List
import cv2
import numpy as np
import pytesseract
import re

from llm_engineering.domains.figure_documents import Figure, Caption, ExtractedFigure


def match_figures_with_captions(figures: List[Figure], captions: List[Caption]) -> List[Figure]:
    matched_figures = []

    for figure in figures:
        fig_center_y = figure.bbox.center_y

        best_caption = None
        min_dist = float("inf")

        for caption in captions:
            cap_center_y = caption.bbox.center_y

            if cap_center_y < fig_center_y:
                continue

            dist = abs(cap_center_y - fig_center_y)
            if dist < min_dist:
                min_dist = dist
                best_caption = caption

        matched_figures.append(Figure(
            bbox=figure.bbox,
            caption=best_caption,
            page=figure.page
        ))

    return matched_figures


def clean_caption(text: str, has_dot: bool = True) -> str:
    """
    Clean caption text.
    - 'Hình X.Y' -> 'H.X.Y'
    - If has_dot=True: 'Hình XY' -> 'H.X.Y' (e.g., Hình 54 -> H.5.4)
    - If has_dot=False: 'Hình XY' -> 'H.XY' (e.g., Hình 54 -> H.54, keep as is)
    - Remove extra text after the number
    """
    if not text:
        return ""

    text = re.sub(r"[Hh]ình\s*", "H.", text, flags=re.IGNORECASE)
    
    # Insert dot between digits only if has_dot=True
    if has_dot:
        text = re.sub(r"H\.(\d)(\d+)", r"H.\1.\2", text, flags=re.IGNORECASE)

    # Extract H.X.Y pattern and remove extra text
    match = re.search(r"(H\.\d+\.?\d*[a-z]?)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1)

    return text.strip()


def extract_caption_text(image: np.ndarray, lang: str = "vie", has_dot: bool = True) -> str:
    text = pytesseract.image_to_string(image, lang=lang)
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = clean_caption(text, has_dot=has_dot)
    return text


def caption_to_filename(text: str, max_len: int = 60) -> str:
    if not text:
        return "figure"

    text = text.lower().strip()
    text = re.sub(r"(?<!\d)\.(?!\d)", "", text)
    text = re.sub(
        r"[^a-z0-9àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹ.\s]",
        "",
        text
    )
    text = re.sub(r"\s+", "_", text)
    return text[:max_len] if text else "figure"


def save_figure_crops(
    matched_figures: List[Figure],
    page_img: np.ndarray,
    output_dir: Path,
    page_idx: int
) -> List[ExtractedFigure]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extracted = []

    for idx, figure in enumerate(matched_figures):
        figure_crop = figure.bbox.crop(page_img)

        if figure.caption and figure.caption.text:
            filename = caption_to_filename(figure.caption.text)
        else:
            filename = f"page{page_idx+1}_fig{idx}"

        save_path = output_dir / f"{filename}.png"

        counter = 1
        while save_path.exists():
            save_path = output_dir / f"{filename}_{counter}.png"
            counter += 1

        cv2.imwrite(str(save_path), figure_crop)

        result = ExtractedFigure.from_figure(figure, save_path)
        extracted.append(result)

    return extracted