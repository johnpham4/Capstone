from pathlib import Path
from typing import List, Union, BinaryIO
import fitz
import gc


def pdf_to_images(
    pdf_source: Union[str, Path, bytes, BinaryIO],
    output_dir: Union[str, Path],
    dpi: int = 200
) -> List[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    doc = None
    try:
        if isinstance(pdf_source, bytes):
            doc = fitz.open(stream=pdf_source, filetype="pdf")
        elif hasattr(pdf_source, 'read'):
            doc = fitz.open(stream=pdf_source.read(), filetype="pdf")
        else:
            doc = fitz.open(str(pdf_source))

        image_paths = []

        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=dpi)
            img_path = output_dir / f"page_{i+1:03d}.png"
            pix.save(str(img_path))
            image_paths.append(img_path)
            pix = None

        return image_paths
    finally:
        if doc:
            doc.close()
        gc.collect()


def cleanup_temp_images(image_paths: List[Path]) -> None:
    for img_path in image_paths:
        if img_path.exists():
            img_path.unlink()
    gc.collect()
