import fitz
import base64
import os
from pathlib import Path
from PIL import Image, ImageEnhance
from typing import List, Dict
from pydantic import BaseModel, Field


class PageBlock(BaseModel):
    """Block detected by YOLO"""
    block_type: str  # "text", "picture", "table", etc.
    bbox: List[float]  # [x1, y1, x2, y2]
    content: str = ""  # Text content or base64 image


class PDFPageExtractor(BaseModel):
    """Extract 1 page from PDF with YOLO layout detection"""

    pdf_path: str
    page_num: int
    zoom: int = Field(default=4, description="Zoom factor for quality")

    class Config:
        arbitrary_types_allowed = True

    def render_page_to_image(self, output_dir: str = "./tmp") -> str:
        """Render PDF page to high-quality image"""
        os.makedirs(output_dir, exist_ok=True)

        doc = fitz.open(self.pdf_path)
        page = doc[self.page_num]

        matrix = fitz.Matrix(self.zoom, self.zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False, colorspace=fitz.csRGB)

        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img = self._enhance_image(img)

        # Resize back to original
        original_size = (int(page.rect.width), int(page.rect.height))
        img = img.resize(original_size, Image.Resampling.LANCZOS)

        temp_img_path = os.path.join(output_dir, f"page_{self.page_num}.png")
        img.save(temp_img_path, "PNG", quality=100, dpi=(300, 300))

        doc.close()
        return temp_img_path

    def _enhance_image(self, img: Image.Image) -> Image.Image:
        """Enhance image quality"""
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.5)

        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.2)

        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.1)

        return img

    def detect_blocks_with_yolo(self, yolo_model, page_image_path: str) -> List[Dict]:
        """Use YOLO to detect layout blocks"""
        results = yolo_model.detect(page_image_path)

        blocks = []
        for result in results:
            boxes = result.boxes.xyxy.numpy()
            classes = result.boxes.cls.numpy()

            for bbox, cls in zip(boxes, classes):
                block_type = self._get_block_type(int(cls))
                blocks.append({
                    "bbox": bbox.tolist(),
                    "type": block_type,
                    "cls": int(cls)
                })

        return blocks

    def _get_block_type(self, class_id: int) -> str:
        """Map YOLO class ID to block type"""
        # DocLayNet classes
        class_map = {
            0: "caption",
            1: "footnote",
            2: "formula",
            3: "list",
            4: "page_footer",
            5: "page_header",
            6: "picture",
            7: "section_header",
            8: "table",
            9: "text",
            10: "title"
        }
        return class_map.get(class_id, "unknown")

    def crop_and_extract_blocks(self, blocks: List[Dict], output_dir: str = "./tmp") -> List[PageBlock]:
        """Crop blocks and extract content (text or image)"""
        doc = fitz.open(self.pdf_path)
        page = doc[self.page_num]

        matrix = fitz.Matrix(self.zoom, self.zoom)
        page_blocks = []

        for idx, block in enumerate(blocks):
            bbox = block["bbox"]
            block_type = block["type"]

            # Crop block
            rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
            pix = page.get_pixmap(clip=rect, matrix=matrix, alpha=False, colorspace=fitz.csRGB)

            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img = self._enhance_image(img)

            # If picture block → save as base64
            if block_type == "picture":
                import io
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                img_base64 = base64.b64encode(buffer.getvalue()).decode()

                page_blocks.append(PageBlock(
                    block_type="picture",
                    bbox=bbox,
                    content=img_base64
                ))
            else:
                # Text block → OCR or extract text
                # For now, save image path for later OCR
                crop_path = os.path.join(output_dir, f"page{self.page_num}_block{idx}.png")
                img.save(crop_path, "PNG", quality=100, dpi=(300, 300))

                page_blocks.append(PageBlock(
                    block_type=block_type,
                    bbox=bbox,
                    content=crop_path  # Will OCR later
                ))

        doc.close()
        return page_blocks
