from __future__ import annotations

import re
import unicodedata


QUESTION_MARKERS = [
    "cmr",
    "c.m.r",
    "chung minh",
    "chung minh rang",
    "chung to",
    "hay chung minh",
    "hay tinh",
    "hay tim",
    "yeu cau",
    "cau hoi",
    "hoi",
    "tinh",
    "tim",
    "ket luan",
    "suy ra",
]


_WS_RE = re.compile(r"\s+")
_DE_BAI_RE = re.compile(r"\bde\s*bai\s*:\s*")
_DSL_RE = re.compile(r"\bdsl\s*:\s*")


def _strip_accents(text: str) -> str:
    text = str(text).replace("đ", "d").replace("Đ", "D")
    return "".join(
        ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn"
    )


def _normalize_for_index(text: str) -> str:
    # Keep index alignment stable for cut operations on the original string.
    return _strip_accents(str(text)).lower()


def _normalize_tail(text: str) -> str:
    return _WS_RE.sub(" ", str(text)).strip().rstrip(" \t:;,-.")


def _find_first_marker_idx(norm_text: str) -> int | None:
    best_idx: int | None = None
    for marker in QUESTION_MARKERS:
        match = re.search(rf"(?<!\\w){re.escape(marker)}\\b", norm_text)
        if match:
            if best_idx is None or match.start() < best_idx:
                best_idx = match.start()
    return best_idx


def _find_enumerated_clause_idx(norm_text: str) -> int | None:
    alpha_matches = list(re.finditer(r"(?:^|[.;:]\\s+)([a-z])\\s*[\\.):]\\s*", norm_text))
    if len(alpha_matches) >= 2 and alpha_matches[0].group(1) == "a":
        return alpha_matches[0].start(1)

    num_matches = list(re.finditer(r"(?:^|[.;:]\\s+)(\\d{1,2})\\s*[\\.):]\\s*", norm_text))
    if len(num_matches) >= 2 and num_matches[0].group(1) == "1":
        return num_matches[0].start(1)

    return None


def _remove_backward_to_dot(text: str, q_idx: int) -> str:
    left_dot = text.rfind(".", 0, q_idx)
    cut_start = left_dot + 1 if left_dot >= 0 else 0

    kept_left = text[:cut_start].rstrip()
    kept_right = text[q_idx + 1 :].lstrip()
    merged = f"{kept_left} {kept_right}".strip()
    return _WS_RE.sub(" ", merged).strip()


def remove_question_part(problem: str) -> str:
    raw = str(problem or "").strip()
    if not raw:
        return ""

    one_line = _WS_RE.sub(" ", raw).strip()
    if not one_line:
        return ""

    norm = _normalize_for_index(one_line)
    marker_idx = _find_first_marker_idx(norm)
    enum_idx = _find_enumerated_clause_idx(norm)
    if enum_idx is not None and (marker_idx is None or enum_idx < marker_idx):
        marker_idx = enum_idx

    q_idx = norm.find("?")
    if q_idx >= 0 and (marker_idx is None or q_idx < marker_idx):
        one_line = _remove_backward_to_dot(one_line, q_idx)
        norm = _normalize_for_index(one_line)
        marker_idx = _find_first_marker_idx(norm)
        enum_idx = _find_enumerated_clause_idx(norm)
        if enum_idx is not None and (marker_idx is None or enum_idx < marker_idx):
            marker_idx = enum_idx

    if marker_idx is None or marker_idx <= 0:
        return _normalize_tail(one_line)

    kept = one_line[:marker_idx].rstrip(" \t:;,-.")
    kept = re.sub(r"([.;:-]\\s*[a-z0-9]{1,3}\\))\\s*$", "", kept, flags=re.IGNORECASE).strip()
    return _normalize_tail(kept)


def prepare_problem_for_dsl(problem: str) -> str:
    raw = str(problem or "").strip()
    if not raw:
        return ""

    cleaned = remove_question_part(raw)
    return cleaned if cleaned else raw


def clean_problem_section(prompt_text: str) -> tuple[str, str]:
    raw_prompt = str(prompt_text or "").strip()
    if not raw_prompt:
        return "", ""

    norm_prompt = _normalize_for_index(raw_prompt)
    de_bai_match = _DE_BAI_RE.search(norm_prompt)
    if not de_bai_match:
        cleaned = prepare_problem_for_dsl(raw_prompt)
        return cleaned, cleaned

    prefix = raw_prompt[: de_bai_match.end()].rstrip()
    after_header = raw_prompt[de_bai_match.end() :]

    norm_after = _normalize_for_index(after_header)
    dsl_match = _DSL_RE.search(norm_after)
    if dsl_match:
        problem_text = after_header[: dsl_match.start()].strip()
        suffix = after_header[dsl_match.start() :].strip()
        cleaned_problem = prepare_problem_for_dsl(problem_text)
        rebuilt = f"{prefix}\n{cleaned_problem}\n\n{suffix}".strip()
        return rebuilt, cleaned_problem

    problem_text = after_header.strip()
    cleaned_problem = prepare_problem_for_dsl(problem_text)
    rebuilt = f"{prefix}\n{cleaned_problem}".strip()
    return rebuilt, cleaned_problem
