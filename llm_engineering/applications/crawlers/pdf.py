import requests
import time
import cv2
import json
import re
from pathlib import Path
from loguru import logger

from llm_engineering.applications.crawlers.base import BaseCrawler
from llm_engineering.applications.networks.figure_detector import FigureDetector
from llm_engineering.applications.preprocessing.operations import (
    pdf_to_images,
    match_figures_with_captions,
    extract_caption_text,
    save_figure_crops,
    cleanup_temp_images,
    ocr_page_text,
    extract_problems_with_figures,
    map_problems_to_figures,
    clean_text,
    has_geometry_keywords
)
from llm_engineering.settings import settings


class PDFCrawler(BaseCrawler):

    def __init__(self):
        super().__init__()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/pdf,*/*',
        })
        self.detector = None

    def _init_detector(self):
        if not self.detector:
            model_path = Path(settings.FIGURE_DETECTOR_LOCAL_DIR)
            self.detector = FigureDetector(
                str(model_path),
                device=settings.FIGURE_DETECTOR_DEVICE,
                model_id=settings.FIGURE_DETECTOR_MODEL
            )

    def extract(self, link: str, **kwargs) -> None:
        output_dir = kwargs.get("output_dir", "dataset/geometry_figures/default")
        name = kwargs.get("name", "unknown")
        has_dot = kwargs.get("has_dot", True)

        logger.info(f"Extracting figures from PDF [{name}] (has_dot={has_dot}): {link}")
        time.sleep(1)

        try:
            response = self.session.get(link, stream=True, timeout=30)
            response.raise_for_status()
            pdf_bytes = response.content

            logger.info(f"Downloaded PDF: {len(pdf_bytes)} bytes")

            self._init_detector()

            base_output_dir = Path(output_dir)
            pages_dir = base_output_dir / "pages"
            figures_dir = base_output_dir / "figures"

            page_images = pdf_to_images(pdf_bytes, pages_dir)

            all_extracted = []
            full_text = ""
            page_text_mapping = {}  # {page_num: text}

            try:
                logger.info(f"Starting OCR for {len(page_images)} pages...")

                # Pass 1: Extract figures + OCR all text
                for page_idx, page_path in enumerate(page_images):
                    page_num = page_idx + 1

                    # Extract figures
                    figures, captions = self.detector(
                        page_path,
                        conf=settings.FIGURE_DETECTION_CONF,
                        imgsz=settings.FIGURE_DETECTION_IMGSZ
                    )

                    for fig in figures:
                        fig.page = page_num

                    page_img = cv2.imread(str(page_path))
                    for caption in captions:
                        caption_crop = caption.bbox.crop(page_img)
                        caption.text = extract_caption_text(caption_crop, lang="vie", has_dot=has_dot)

                    matched_figures = match_figures_with_captions(figures, captions)

                    extracted = save_figure_crops(
                        matched_figures=matched_figures,
                        page_img=page_img,
                        output_dir=figures_dir,
                        page_idx=page_idx
                    )

                    all_extracted.extend([fig.to_dict() for fig in extracted])

                    # Two-pass OCR: Quick scan for geometry keywords, then high-quality OCR
                    # Only OCR pages with "Hình" or "Giác" (geometry-related)
                    if has_geometry_keywords(page_path):
                        page_text = ocr_page_text(page_path)
                        page_text_mapping[page_num] = page_text
                        full_text += f"\n===PAGE_{page_num}===\n{page_text}\n"
                        logger.info(f"OCR page {page_num} (geometry keywords detected)")
                    else:
                        page_text_mapping[page_num] = ""

                    if page_num % 20 == 0:
                        logger.info(f"Processed {page_num}/{len(page_images)} pages")

                    page_img = None

                logger.info(f"OCR completed. Total text: {len(full_text)} chars")

                # Export full text for debugging
                debug_file = base_output_dir / f"{name}_full_ocr_text.txt"
                with open(debug_file, 'w', encoding='utf-8') as f:
                    f.write(full_text)
                logger.info(f"Exported full OCR text to {debug_file}")

                # Pass 2: Parse problems from full text
                all_problems = self._parse_geometry_problems(full_text, page_text_mapping)
                logger.info(f"Extracted {len(all_problems)} geometry problems")

                # Filter: filter out the right format of images
                seen_captions, filtered_figures = self._filter_figures(all_extracted, has_dot=has_dot)

                # Map problems to extracted figures
                mapped_problems = map_problems_to_figures(all_problems, filtered_figures)
                matched_count = sum(1 for p in mapped_problems if p.get('match_count', 0) > 0)

                # Save figures JSON
                json_path = base_output_dir / f"{name}_figures.json"
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        "source_pdf": link,
                        "name": name,
                        "total_figures": len(filtered_figures),
                        "figures": filtered_figures
                    }, f, ensure_ascii=False, indent=2)

                logger.info(f"Saved {len(filtered_figures)} figures to {json_path}")

                # Save problems JSON
                problems_json_path = base_output_dir / f"{name}_problems.json"
                with open(problems_json_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        "source_pdf": link,
                        "name": name,
                        "total_problems": len(mapped_problems),
                        "problems_with_solution": sum(1 for p in mapped_problems if p.get('has_solution', False)),
                        "problems_without_solution": sum(1 for p in mapped_problems if not p.get('has_solution', False)),
                        "problems_matched_to_figures": matched_count,
                        "problems": mapped_problems
                    }, f, ensure_ascii=False, indent=2)

                logger.info(f"Saved {len(mapped_problems)} problems ({matched_count} matched to figures) to {problems_json_path}")

            finally:
                cleanup_temp_images(page_images)

        except Exception as e:
            logger.error(f"Error crawling PDF {link}: {e}")
            raise

    def _delete_figure_image(self, image_path: str) -> None:
        img_path = Path(image_path)
        if img_path.exists():
            img_path.unlink()

    def _is_valid_caption(self, caption: str, has_dot: bool) -> bool:
        if not caption or not caption.startswith('H.'):
            return False

        if has_dot:
            return bool(re.match(r'^H\.\d+\.\d+[a-zA-Z]?$', caption))
        else:
            return bool(re.match(r'^H\.\d+$', caption))

    def _filter_figures(self, all_extracted: list, has_dot: bool = True) -> tuple[set, list]:
        seen_captions = set()
        filtered_figures = []

        for fig in all_extracted:
            caption = fig['caption']

            # Validate caption format
            if not self._is_valid_caption(caption, has_dot):
                self._delete_figure_image(fig['image_path'])
                continue

            # Skip duplicates (keep first occurrence)
            if caption in seen_captions:
                self._delete_figure_image(fig['image_path'])
                continue

            seen_captions.add(caption)
            filtered_figures.append(fig)

        removed = len(all_extracted) - len(filtered_figures)
        logger.info(f"Filtered {len(all_extracted)} -> {len(filtered_figures)} figures (removed {removed} invalid/duplicates)")
        return seen_captions, filtered_figures

    def _parse_geometry_problems(self, full_text: str, page_text_mapping: dict) -> list:
        """
        Parse geometry problems from full PDF text.

        Logic:
        1. Tách theo số bài (1., 2., 3., ...)
        2. Detect phần đề vs phần giải (dựa vào "HƯỚNG DẪN GIẢI")
        3. Ghép bài có cùng số (4. đề + 4. giải)
        4. CHỈ giữ bài giải có (H.xx) hoặc Hình → bài hình học
        5. Loại bỏ toán đại số
        """
        # Find solution section marker
        solution_start_pos = -1
        markers = ['LỜI GIẢI', 'HƯỚNG DẪN', 'ĐÁP SỐ', 'HƯỚNG DẪN GIẢI', 'ĐÁP ÁN', 'BÀI GIẢI']
        for marker in markers:
            pos = full_text.upper().find(marker.upper())
            if pos > 0:
                solution_start_pos = pos
                logger.info(f"Found solution section '{marker}' at position {pos}")
                break

        if solution_start_pos < 0:
            logger.warning("Could not find solution section marker")
            return []

        # Pattern: số thứ tự bài
        problem_pattern = r'(\d+)\.\s+'

        # Find all problem numbers
        matches = list(re.finditer(problem_pattern, full_text))
        logger.info(f"Found {len(matches)} problem numbers")

        # Separate questions and solutions
        questions = {}  # {num: {'text': ..., 'page': ...}}
        solutions = {}  # {num: {'text': ..., 'page': ...}}

        for i, match in enumerate(matches):
            prob_num = match.group(1)
            start = match.end()

            # Get segment until next problem
            if i + 1 < len(matches):
                end = matches[i + 1].start()
            else:
                end = len(full_text)

            segment = full_text[start:end].strip()

            # Determine page number
            page_num = 1
            for pg, pg_text in page_text_mapping.items():
                if segment[:100] in pg_text:
                    page_num = pg
                    break

            # Is this question or solution?
            if match.start() < solution_start_pos:
                # Question
                questions[prob_num] = {
                    'text': segment,
                    'page': page_num
                }
            else:
                # Solution
                solutions[prob_num] = {
                    'text': segment,
                    'page': page_num
                }

        logger.info(f"Parsed {len(questions)} questions, {len(solutions)} solutions")

        # Filter: CHỈ giữ bài có figure references trong QUESTION (đề bài)
        # Loại bỏ toán đại số (không đề cập hình vẽ)
        geometry_problems = []
        figure_pattern = r'\([Hh]\.?\d+\.?\d*\)|[Hh]ình\s*\d+\.?\d*'

        for prob_num in questions.keys():
            if prob_num not in solutions:
                continue

            question_text = questions[prob_num]['text']
            solution_text = solutions[prob_num]['text']

            # Clean and normalize text
            question_cleaned = clean_text(question_text)
            solution_cleaned = clean_text(solution_text)

            # Check if QUESTION mentions figures (geometry problem)
            # This filters out algebra problems
            if re.search(figure_pattern, question_cleaned, re.IGNORECASE):
                # Extract figure references from cleaned question text
                fig_refs = re.findall(figure_pattern, question_cleaned, re.IGNORECASE)

                geometry_problems.append({
                    'problem_number': prob_num,
                    'question': question_cleaned,
                    'solution': solution_cleaned,
                    'page_number': questions[prob_num]['page'],
                    'solution_page': solutions[prob_num]['page'],
                    'figure_references': list(set(fig_refs)),
                    'has_solution': True
                })

        logger.info(f"Filtered to {len(geometry_problems)} geometry problems (with figures)")
        return geometry_problems