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
_ENUM_RE = re.compile(
    r"(?:^|[^a-z0-9])(?P<label>[a-z]|\d{1,2})\s*[\.\):]\s*",
    flags=re.IGNORECASE,
)
_TRAILING_ENUM_RE = re.compile(
    r"(?:[;,:-]?\s*(?:[a-z]|\d{1,2})\s*[\.\):]\s*)+$",
    flags=re.IGNORECASE,
)


def _strip_accents(text: str) -> str:
    # Keep a 1:1 mapping between normalized and original text so cut indices are
    # still valid on the source string.
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
    return _strip_accents(str(text)).lower()


def _normalize_tail(text: str) -> str:
    return _WS_RE.sub(" ", str(text)).strip().rstrip(" \t:;,-.")


def _find_first_marker_idx(norm_text: str) -> int | None:
    best_idx: int | None = None
    for marker in QUESTION_MARKERS:
        marker_norm = _normalize_for_index(marker)
        match = re.search(rf"(?<!\w){re.escape(marker_norm)}\b", norm_text)
        if match:
            if best_idx is None or match.start() < best_idx:
                best_idx = match.start()
    return best_idx


def _find_enumerated_clause_idx(norm_text: str) -> int | None:
    # Detect "a) ... b) ..." and "1) ... 2) ..." style sub-question blocks.
    matches = list(_ENUM_RE.finditer(norm_text))
    if not matches:
        return None

    alpha = [(m.group("label").lower(), m.start("label")) for m in matches if m.group("label").isalpha()]
    if alpha:
        first_a = next((idx for label, idx in alpha if label == "a"), None)
        if first_a is not None and any(label == "b" and idx > first_a for label, idx in alpha):
            return first_a

    numeric = [(m.group("label"), m.start("label")) for m in matches if m.group("label").isdigit()]
    if numeric:
        first_1 = next((idx for label, idx in numeric if label == "1"), None)
        if first_1 is not None and any(label == "2" and idx > first_1 for label, idx in numeric):
            return first_1

    return None


def _remove_backward_to_dot_once(text: str, q_idx: int) -> str:
    left_dot = text.rfind(".", 0, q_idx)
    cut_start = left_dot + 1 if left_dot >= 0 else 0

    kept_left = text[:cut_start].rstrip()
    kept_right = text[q_idx + 1 :].lstrip()
    merged = f"{kept_left} {kept_right}".strip()
    return _WS_RE.sub(" ", merged).strip()


def _remove_question_mark_segments(text: str) -> str:
    result = str(text)
    while True:
        q_idx = result.find("?")
        if q_idx < 0:
            break
        result = _remove_backward_to_dot_once(result, q_idx)
    return _normalize_tail(result)


def remove_question_part(problem: str) -> str:
    raw = str(problem or "").strip()
    if not raw:
        return ""

    one_line = _WS_RE.sub(" ", raw).strip()
    if not one_line:
        return ""

    if "?" in one_line:
        one_line = _remove_question_mark_segments(one_line)
        if not one_line:
            return ""

    norm = _normalize_for_index(one_line)
    marker_idx = _find_first_marker_idx(norm)
    enum_idx = _find_enumerated_clause_idx(norm)

    cut_candidates = [idx for idx in (marker_idx, enum_idx) if idx is not None]
    if not cut_candidates:
        return _normalize_tail(one_line)

    cut_idx = min(cut_candidates)
    if cut_idx <= 0:
        return ""

    kept = one_line[:cut_idx].rstrip(" \t:;,-.")
    kept = _TRAILING_ENUM_RE.sub("", kept).strip()
    return _normalize_tail(kept)


def prepare_problem_for_dsl(problem: str) -> str:
    raw = str(problem or "").strip()
    if not raw:
        return ""

    return remove_question_part(raw)


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