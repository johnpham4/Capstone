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
    # Preserve a 1:1 character mapping so indices in normalized text align with the
    # original string. For each original character, return a single base character
    # (e.g., 'á' -> 'a', 'đ' -> 'd') so slicing using indices found on the
    # normalized string can be applied to the original string.
    s = str(text)
    out_chars: list[str] = []
    for ch in s:
        if ch == "đ":
            out_chars.append("d")
            continue
        if ch == "Đ":
            out_chars.append("D")
            continue
        decomp = unicodedata.normalize("NFD", ch)
        base = "".join(c for c in decomp if unicodedata.category(c) != "Mn")
        if base:
            out_chars.append(base[0])
        else:
            out_chars.append(ch)
    return "".join(out_chars)


def _normalize_for_index(text: str) -> str:
    # Keep index alignment stable for cut operations on the original string.
    # Map each original character to a single base character so indices remain
    # aligned between the normalized and original strings.
    return "".join(_strip_accents(ch).lower() for ch in str(text))


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
    # Detect enumerated sub-questions such as "a) ... b) ..." or "1) ... 2) ...".
    # Use patterns that match a standalone letter/number followed by ., ) or :.
    # Match enumerators that are preceded by start-of-string or a non-alphanumeric
    # character. This avoids reliance on lookbehind and is robust after we
    # collapsed whitespace into single spaces.
    alpha_matches = list(
        re.finditer(r"(?:^|[^A-Za-z0-9])([a-z])\s*[\.\):]\s*", norm_text, flags=re.IGNORECASE | re.MULTILINE)
    )
    if len(alpha_matches) >= 2 and alpha_matches[0].group(1).lower() == "a":
        return alpha_matches[0].start(1)

    num_matches = list(
        re.finditer(r"(?:^|[^A-Za-z0-9])(\d{1,2})\s*[\.\):]\s*", norm_text, flags=re.MULTILINE)
    )
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
    kept = re.sub(r"([.;:-]\s*[a-z0-9]{1,3}\))\s*$", "", kept, flags=re.IGNORECASE).strip()
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
