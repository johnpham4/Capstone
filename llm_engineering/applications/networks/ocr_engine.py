"""OCR Engine for text extraction with singleton pattern."""

import re
import cv2
import pytesseract
from pathlib import Path
from typing import Optional
from paddleocr import PaddleOCR
from loguru import logger


class OCREngine:
    """Singleton OCR engine supporting both Tesseract (fast) and PaddleOCR (accurate)."""

    _paddle_instance: Optional[PaddleOCR] = None

    def __init__(self):
        """Initialize OCR engine."""
        pass

    @classmethod
    def get_paddle_instance(cls, lang: str = "vi") -> PaddleOCR:
        """Get or create PaddleOCR singleton instance."""
        if cls._paddle_instance is None:
            logger.info("Initializing PaddleOCR for Vietnamese...")
            cls._paddle_instance = PaddleOCR(
                use_angle_cls=True,  # Enable text angle detection
                lang=lang,           # Vietnamese
            )
            logger.info("PaddleOCR initialized successfully")
        return cls._paddle_instance

    def quick_scan_keywords(self, image_path: str | Path, keywords: list[str]) -> bool:
        """Quick Tesseract scan to check if page contains keywords.

        Args:
            image_path: Path to page image
            keywords: List of regex patterns to search for

        Returns:
            True if any keyword found
        """
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return False

            # Fast OCR with Tesseract
            text = pytesseract.image_to_string(img, lang="vie", config="--psm 6")

            # Check for any keyword
            for keyword in keywords:
                if re.search(keyword, text, re.IGNORECASE):
                    return True
            return False

        except Exception as e:
            logger.warning(f"Keyword detection failed for {image_path}: {e}")
            return False

    def ocr_high_quality(self, image_path: str | Path, lang: str = "vi") -> str:
        """High-quality OCR using PaddleOCR.

        Args:
            image_path: Path to page image
            lang: OCR language (default: Vietnamese 'vi')

        Returns:
            Extracted text preserving spatial layout
        """
        img = cv2.imread(str(image_path))
        if img is None:
            return ""

        try:
            ocr = self.get_paddle_instance(lang)
            result = ocr.ocr(img)

            if not result or not result[0]:
                return ""

            # Extract and sort text blocks by position
            text_blocks = []
            for line in result[0]:
                if not line or len(line) < 2:
                    continue

                box = line[0]
                text_info = line[1]

                if not box or not text_info or len(text_info) < 2:
                    continue

                text = text_info[0]
                confidence = text_info[1]

                # Skip low confidence (< 0.5)
                if confidence < 0.5:
                    continue

                y_pos = min([point[1] for point in box])
                x_pos = min([point[0] for point in box])
                text_blocks.append((y_pos, x_pos, text, confidence))

            if not text_blocks:
                return ""

            # Sort by Y (row), then X (column)
            text_blocks.sort(key=lambda x: (x[0] // 20, x[1]))

            # Build text with line breaks
            lines = []
            current_y = -1
            current_line = []

            for y_pos, x_pos, text, conf in text_blocks:
                y_bucket = int(y_pos // 20)

                if current_y == -1:
                    current_y = y_bucket

                if y_bucket != current_y:
                    if current_line:
                        lines.append(' '.join(current_line))
                    current_line = [text]
                    current_y = y_bucket
                else:
                    current_line.append(text)

            if current_line:
                lines.append(' '.join(current_line))

            return '\n'.join(lines)

        except Exception as e:
            logger.error(f"OCR failed for {image_path}: {e}")
            return ""
