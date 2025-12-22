# import re
# import cv2
# import pytesseract
# from pathlib import Path
# from paddleocr import PaddleOCR
# from loguru import logger
# import paddle

# from llm_engineering.applications.networks.base import SingletonMeta


# class OCREngine(metaclass=SingletonMeta):
#     def __init__(self):

#         paddle.set_device("gpu" if paddle.is_compiled_with_cuda() else "cpu")

#         self.model = PaddleOCR(
#             use_angle_cls=False,  # Disable angle detection for 2x speed boost
#             lang="vi",
#             det_db_box_thresh=0.5, # Higher threshold = faster detection
#             rec_batch_num=5,
#         )


#     def quick_scan_keywords(self, image_path: str | Path, keywords: list[str]) -> bool:
#         """Quick Tesseract scan to check if page contains keywords.

#         Args:
#             image_path: Path to page image
#             keywords: List of regex patterns to search for

#         Returns:
#             True if any keyword found
#         """
#         try:
#             img = cv2.imread(str(image_path))
#             if img is None:
#                 return False

#             # Fast OCR with Tesseract
#             text = pytesseract.image_to_string(img, lang="vie", config="--psm 6")

#             # Check for any keyword
#             for keyword in keywords:
#                 if re.search(keyword, text, re.IGNORECASE):
#                     return True
#             return False

#         except Exception as e:
#             logger.warning(f"Keyword detection failed for {image_path}: {e}")
#             return False

#     def ocr_high_quality(self, image_path: str | Path) -> str:
#         """High-quality OCR using PaddleOCR.

#         Args:
#             image_path: Path to page image

#         Returns:
#             Extracted text preserving spatial layout
#         """
#         img = cv2.imread(str(image_path))
#         if img is None:
#             return ""

#         try:
#             result = self.model.ocr(img)

#             if not result or not result[0]:
#                 return ""

#             # Extract text blocks with position sorting
#             text_blocks = []
#             for line in result[0]:
#                 if not line or len(line) < 2:
#                     continue

#                 box, text_info = line[0], line[1]
#                 if not box or not text_info or len(text_info) < 2:
#                     continue

#                 text, confidence = text_info[0], text_info[1]

#                 # Skip low confidence
#                 if confidence < 0.5:
#                     continue

#                 y_pos = min(point[1] for point in box)
#                 x_pos = min(point[0] for point in box)
#                 text_blocks.append((y_pos, x_pos, text, confidence))

#             if not text_blocks:
#                 return ""

#             # Sort by row (Y), then column (X)
#             text_blocks.sort(key=lambda x: (x[0] // 20, x[1]))

#             # Group text by rows
#             lines, current_line = [], []
#             current_y = -1

#             for y_pos, x_pos, text, _ in text_blocks:
#                 y_bucket = y_pos // 20

#                 if current_y == -1:
#                     current_y = y_bucket

#                 if y_bucket != current_y:
#                     if current_line:
#                         lines.append(' '.join(current_line))
#                     current_line = [text]
#                     current_y = y_bucket
#                 else:
#                     current_line.append(text)

#             if current_line:
#                 lines.append(' '.join(current_line))

#             return '\n'.join(lines)

#         except Exception as e:
#             logger.error(f"OCR failed for {image_path}: {e}")
#             return ""
