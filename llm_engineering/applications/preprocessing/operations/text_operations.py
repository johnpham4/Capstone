import re
from typing import List, Dict
from pathlib import Path
from loguru import logger

from llm_engineering.applications.networks.ocr_engine import OCREngine
from llm_engineering.settings import settings

_ocr_engine = OCREngine()


def has_geometry_keywords(image_path: str | Path) -> bool:
    """Check if page contains geometry keywords (Hình, Giác)."""
    return _ocr_engine.quick_scan_keywords(
        image_path,
        keywords=[r'[Hh]ình', r'[Gg]iác']
    )


def ocr_page_text(image_path: str | Path, lang: str = "vi") -> str:
    """Extract text using high-quality PaddleOCR."""
    return _ocr_engine.ocr_high_quality(image_path, lang)


def extract_problems_with_figures(text: str) -> List[Dict]:
    # Pattern tìm reference đến hình vẽ
    figure_patterns = [
        (r'Hình\s*\d+[.\s]*\d+', 'Hình X.Y'),
        (r'\(H\.?\d+(?:\.\d+)?\)', '(H.X.Y)'),
        (r'trong\s+(?:các\s+)?hình\s+(?:vẽ\s+)?(?:sau|trên)', 'trong hình...'),
    ]

    results = []

    # Strategy 1: Tìm các vị trí số thứ tự bài tập (X.Y..)
    problem_number_pattern = r'(\d+\.\d+)\.\.'
    problem_starts = list(re.finditer(problem_number_pattern, text))

    if problem_starts:
        for i, match in enumerate(problem_starts):
            start_pos = match.start()
            end_pos = problem_starts[i+1].start() if i+1 < len(problem_starts) else len(text)

            segment = text[start_pos:end_pos].strip()
            problem_num = match.group(1)

            # Kiểm tra xem segment có đề cập hình không
            found_figures = []
            for fig_pattern, _ in figure_patterns:
                matches = re.findall(fig_pattern, segment, re.IGNORECASE)
                found_figures.extend(matches)

            if found_figures:
                # Tìm xem có "Giải" trong segment không
                giai_match = re.search(r'Giải', segment, re.IGNORECASE)

                if giai_match:
                    question_text = segment[:giai_match.start()].strip()
                    solution_text = segment[giai_match.end():].strip()
                else:
                    question_text = segment
                    solution_text = ""

                # Clean up question
                question_text = re.sub(r'^[-@•()\s=]+', '', question_text).strip()
                question_text = re.sub(r'^\d+\.\d+\.\.', '', question_text).strip()

                results.append({
                    'problem_number': problem_num,
                    'content': segment,
                    'question': question_text,
                    'solution': solution_text,
                    'figure_references': list(set(found_figures)),
                    'has_solution': bool(giai_match)
                })

    # Strategy 2: Nếu không có pattern số thứ tự, tìm "Giải"
    else:
        giai_matches = list(re.finditer(r'Giải', text, re.IGNORECASE))

        if giai_matches:
            problem_num = 0
            for giai_match in giai_matches:
                giai_pos = giai_match.start()

                search_start = 0
                for prev_match in giai_matches:
                    if prev_match.start() < giai_pos:
                        search_start = prev_match.end()

                question_segment = text[search_start:giai_pos].strip()

                # Kiểm tra xem có pattern hình nào không
                found_figures = []
                for fig_pattern, _ in figure_patterns:
                    matches = re.findall(fig_pattern, question_segment, re.IGNORECASE)
                    found_figures.extend(matches)

                if found_figures:
                    problem_num += 1

                    solution_end = len(text)
                    for next_match in giai_matches:
                        if next_match.start() > giai_pos:
                            solution_end = next_match.start()
                            break

                    solution_text = text[giai_match.end():solution_end].strip()
                    question_text = re.sub(r'^[-@•()\s=]+', '', question_segment).strip()

                    results.append({
                        'problem_number': str(problem_num),
                        'content': text[search_start:solution_end].strip(),
                        'question': question_text,
                        'solution': solution_text,
                        'figure_references': list(set(found_figures)),
                        'has_solution': True
                    })

        # Strategy 3: Tìm tất cả đề cập hình và lấy context
        else:
            all_figure_matches = []
            for fig_pattern, _ in figure_patterns:
                for match in re.finditer(fig_pattern, text, re.IGNORECASE):
                    all_figure_matches.append((match.start(), match.end(), match.group()))

            all_figure_matches.sort(key=lambda x: x[0])

            if all_figure_matches:
                problem_num = 0
                for i, (start, end, fig_ref) in enumerate(all_figure_matches):
                    # Context xung quanh (200 ký tự trước và sau)
                    context_start = max(0, start - 200)
                    context_end = min(len(text), end + 200)

                    segment = text[context_start:context_end].strip()
                    problem_num += 1

                    results.append({
                        'problem_number': str(problem_num),
                        'content': segment,
                        'question': segment,
                        'solution': "",
                        'figure_references': [fig_ref],
                        'has_solution': False
                    })

    return results


def normalize_figure_reference(ref: str) -> str:
    # Remove parentheses and "ình"
    ref = ref.replace('(', '').replace(')', '').replace('Hình', 'H').replace('ình', '')
    ref = ref.strip()

    # Already in H.X.Y format
    if re.match(r'^H\.\d+\.\d+', ref):
        return ref

    # H.XY format -> keep as is
    if re.match(r'^H\.\d+$', ref):
        return ref

    # HXY or H XY -> H.X.Y (if 2 digits) or H.XY (if more)
    match = re.search(r'H\.?\s*(\d+)', ref)
    if match:
        nums = match.group(1)
        if len(nums) == 2:
            return f"H.{nums[0]}.{nums[1]}"
        else:
            return f"H.{nums}"

    return ref


def map_problems_to_figures(
    problems: List[Dict],
    figures_metadata: List[Dict]
) -> List[Dict]:
    # Build figure lookup by caption
    figure_lookup = {}
    for fig in figures_metadata:
        caption = fig.get('caption', '')
        if caption:
            normalized = normalize_figure_reference(caption)
            figure_lookup[normalized] = fig

    mapped = []
    for prob in problems:
        # Normalize all references in problem
        normalized_refs = [
            normalize_figure_reference(ref)
            for ref in prob.get('figure_references', [])
        ]

        # Find matching figures
        matched_figures = []
        for ref in normalized_refs:
            if ref in figure_lookup:
                matched_figures.append({
                    'reference': ref,
                    'figure_path': figure_lookup[ref].get('path'),
                    'page': figure_lookup[ref].get('page'),
                    'caption': figure_lookup[ref].get('caption')
                })

        mapped_prob = {
            **prob,
            'normalized_references': normalized_refs,
            'matched_figures': matched_figures,
            'match_count': len(matched_figures)
        }
        mapped.append(mapped_prob)

    return mapped
