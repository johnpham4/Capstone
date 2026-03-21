import argparse
import glob
import imghdr
import json
import os
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

DEFAULT_JSON_FILES = [
    "dataset/Minh/full_part_001.json",
    "dataset/Minh/full_part_002.json",
    "dataset/Minh/full_part_003.json",
    "dataset/Minh/full_part_004.json",
    "dataset/Minh/full_part_005.json",
]
DEFAULT_IMAGE_ROOT = "dataset/Minh/images"
DEFAULT_OUTPUT_DIR = "dataset/Minh"
DEFAULT_CHUNK_SIZE = 20
DEFAULT_SUMMARY_FILENAME = "review_full_sweep_summary.json"
DEFAULT_PASS_FILENAME = "review_pass.json"
DEFAULT_FAIL_FILENAME = "review_fail.json"
DEFAULT_REVIEW_FILENAME = "review_ambiguous.json"
DEFAULT_IMAGE_MISSING_FILENAME = "review_image_missing.json"


def norm_text(text: str) -> str:
    s = unicodedata.normalize("NFD", text.lower())
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return s


def split_tokens(expr: str) -> List[str]:
    inner = expr.strip()[1:-1].strip() if expr.strip().startswith("(") and expr.strip().endswith(")") else expr
    return [t for t in re.split(r"\s+", inner) if t]


def parse_dsl_lines(answer: str) -> List[str]:
    return [ln.strip() for ln in answer.splitlines() if ln.strip()]


def check_parentheses(answer: str) -> bool:
    bal = 0
    for ch in answer:
        if ch == "(":
            bal += 1
        elif ch == ")":
            bal -= 1
            if bal < 0:
                return False
    return bal == 0


def extract_shapes_vertices(lines: List[str]) -> set:
    vertices = set()
    shape_pat = re.compile(
        r"^\((triangle|quadrilateral|square|rectangle|trapezoid|parallelogram|rhombus)\s+\(([A-Z ]+)\)"
    )
    for ln in lines:
        m = shape_pat.search(ln)
        if m:
            for v in m.group(2).split():
                if len(v) == 1 and v.isalpha() and v.isupper():
                    vertices.add(v)
    return vertices


def extract_angle_requirements(text: str) -> List[Tuple[str, str, str, str]]:
    reqs = []
    # Capture variants such as: goc ABC = 90, ∠ABC = 90
    for m in re.finditer(r"(?:goc|∠)\s*([a-z]{3})\s*=\s*(\d{1,3})", text):
        a, b, c = m.group(1).upper()
        deg = m.group(2)
        reqs.append((a, b, c, deg))
    return reqs


def extract_midpoint_requirements(text: str) -> List[Tuple[str, str, str]]:
    reqs = []
    # Example: D la trung diem cua doan thang AB
    for m in re.finditer(r"\b([a-z])\s+la\s+trung\s+diem\s+cua\s+doan\s+thang\s+([a-z])([a-z])", text):
        reqs.append((m.group(1).upper(), m.group(2).upper(), m.group(3).upper()))
    return reqs


def extract_segment_mentions(text: str) -> List[Tuple[str, str]]:
    reqs = []
    # Only strong mentions to reduce false positives.
    for m in re.finditer(r"(?:doan\s+thang|duong\s+kinh)\s+([a-z])([a-z])", text):
        reqs.append((m.group(1).upper(), m.group(2).upper()))
    return reqs


def is_ambiguous_problem(text: str) -> bool:
    patterns = [
        # "duong thang la ..." with missing object/relation is under-specified.
        r"duong\s+thang\s+la\b",
        # "duong thang di qua diem X" is under-specified unless a second condition exists.
        r"duong\s+thang\s+di\s+qua\s+diem\s+[a-z]\b(?!\s+(va|vuong\s+goc|song\s+song|cat|la))",
        # Generic line mention with no defining relation.
        r"ve\s+duong\s+thang\b(?!\s+(di\s+qua|vuong\s+goc|song\s+song|cat|qua))",
    ]
    return any(re.search(p, text) for p in patterns)


def has_segment(answer: str, a: str, b: str) -> bool:
    p1 = f"(segment {a} {b})"
    p2 = f"(segment {b} {a})"
    return (p1 in answer) or (p2 in answer)


def resolve_image(sample: Dict, json_path: str, image_root: str) -> Optional[str]:
    image_dir = sample.get("image_dir")
    if isinstance(image_dir, str) and image_dir.strip():
        candidate = image_dir.replace("\\", "/")
        if not os.path.isabs(candidate):
            c1 = os.path.normpath(os.path.join(os.path.dirname(json_path), candidate))
            c2 = os.path.normpath(os.path.join(image_root, os.path.basename(candidate)))
            if os.path.exists(c1):
                return c1
            if os.path.exists(c2):
                return c2
        elif os.path.exists(candidate):
            return candidate

        # Fallback: map by numeric id embedded in image_dir, e.g. img_12345 -> diagram_12345
        num_match = re.search(r"(\d+)", candidate)
        if num_match:
            nid = num_match.group(1)
            patterns = [
                os.path.join(image_root, f"*{nid}*.png"),
                os.path.join(image_root, f"*{nid}*.jpg"),
                os.path.join(image_root, f"*{nid}*.jpeg"),
                os.path.join(image_root, f"*{nid}*.webp"),
            ]
            for pat in patterns:
                found = glob.glob(pat)
                if found:
                    return found[0]

    sid = str(sample.get("id", "")).strip()
    if sid:
        patterns = [
            os.path.join(image_root, f"*{sid}*.png"),
            os.path.join(image_root, f"*{sid}*.jpg"),
            os.path.join(image_root, f"*{sid}*.jpeg"),
            os.path.join(image_root, f"*{sid}*.webp"),
        ]
        for pat in patterns:
            found = glob.glob(pat)
            if found:
                return found[0]
    return None


def check_image_file(path: str) -> bool:
    if not os.path.exists(path):
        return False
    if os.path.getsize(path) <= 0:
        return False
    return imghdr.what(path) is not None


def evaluate_sample(sample: Dict, json_path: str, image_root: str) -> Dict:
    sid = str(sample.get("id") or sample.get("image_dir") or "unknown")
    instruction = str(sample.get("instruction", ""))
    problem = str(sample.get("problem", ""))
    answer = str(sample.get("answer", ""))
    base_fields = {
        "image_dir": sample.get("image_dir"),
        "instruction": instruction,
        "answer": answer,
        "problem": problem,
    }

    codes: List[str] = []
    lines = parse_dsl_lines(answer)
    norm_all = norm_text(f"{instruction} {problem}")

    # DSL syntax checks.
    if not check_parentheses(answer):
        codes.append("DSL_SYNTAX")
    for ch in ["⟂", "∥", "∠", "="]:
        if ch in answer:
            codes.append("DSL_SYNTAX")
            break
    if "//" in answer:
        codes.append("DSL_SYNTAX")

    # on-segment arity.
    for ln in lines:
        for m in re.finditer(r"\(on-segment\s+([^\)]*)\)", ln):
            args = [x for x in m.group(1).split() if x]
            if len(args) != 3:
                codes.append("DSL_SYNTAX")

    # Define rules.
    seen_def = set()
    vertices = extract_shapes_vertices(lines)
    for ln in lines:
        dm = re.match(r"^\(define\s+([A-Za-z][A-Za-z0-9_]*)\s+point(\s+\([^\)]*\))?\)$", ln)
        if dm:
            p = dm.group(1)
            if p in seen_def:
                codes.append("DSL_POINT_DEFINE")
            seen_def.add(p)
            if p in vertices and dm.group(2) is None:
                codes.append("DSL_POINT_DEFINE")

    # Incenter/circumcenter consistency.
    if "(incenter " in answer and "(incircle " not in answer:
        codes.append("DSL_MISSING_CONSTRAINT")
    if "(circumcenter " in answer and "(circumcircle " not in answer:
        codes.append("DSL_MISSING_CONSTRAINT")

    # Angle mapping checks.
    for a, b, c, deg in extract_angle_requirements(norm_all):
        if f"(angle-measure {a} {b} {c} {deg})" not in answer:
            codes.append("DSL_ANGLE")
            break

    # Midpoint checks.
    for m, a, b in extract_midpoint_requirements(norm_all):
        cond1 = f"(define {m} point (midpoint {a} {b}))" in answer
        cond2 = f"(define {m} point (midpoint {b} {a}))" in answer
        cond3 = f"(equal-distance {a} {m} {m} {b})" in answer
        cond4 = f"(equal-distance {b} {m} {m} {a})" in answer
        if not (cond1 or cond2 or cond3 or cond4):
            codes.append("DSL_WRONG_MAPPING")
            break

    # Segment mention checks.
    for a, b in extract_segment_mentions(norm_all):
        if not has_segment(answer, a, b):
            codes.append("DSL_MISSING_CONSTRAINT")
            break

    image_path = resolve_image(sample, json_path, image_root)
    if not image_path:
        return {
            **base_fields,
            "id": sid,
            "status": "IMAGE_MISSING",
            "codes": ["IMAGE_MISSING"],
            "short_reason": "Khong tim thay anh theo image_dir/id.",
            "action": "rerender",
        }

    if not check_image_file(image_path):
        return {
            **base_fields,
            "id": os.path.basename(image_path).replace("\\", "/"),
            "status": "FAIL",
            "codes": ["IMAGE_LAYOUT_BAD"],
            "short_reason": "File anh loi hoac khong doc duoc.",
            "action": "rerender",
        }

    resolved_id = os.path.basename(image_path).replace("\\", "/")

    # De-duplicate codes while preserving order.
    dedup_codes = []
    seen = set()
    for c in codes:
        if c not in seen:
            dedup_codes.append(c)
            seen.add(c)

    if dedup_codes:
        return {
            **base_fields,
            "id": resolved_id,
            "status": "FAIL",
            "codes": dedup_codes,
            "short_reason": "DSL co loi theo bo rule prompt_dsl.",
            "action": "fix_dsl",
        }

    if is_ambiguous_problem(norm_all):
        return {
            **base_fields,
            "id": resolved_id,
            "status": "REVIEW",
            "codes": ["AMBIGUOUS_PROBLEM"],
            "short_reason": "De bai mo ho/chua du rang buoc ro rang, can review thu cong.",
            "action": "manual_review",
        }

    return {
        **base_fields,
        "id": resolved_id,
        "status": "PASS",
        "codes": [],
        "short_reason": "DSL hop le va de bai du ro rang theo bo check tu dong.",
        "action": "none",
    }


def severity_for(item: Dict) -> str:
    if item["status"] == "IMAGE_MISSING":
        return "MEDIUM"
    if "DSL_SYNTAX" in item.get("codes", []):
        return "CRITICAL"
    if "DSL_ANGLE" in item.get("codes", []) or "DSL_WRONG_MAPPING" in item.get("codes", []):
        return "HIGH"
    if item["status"] == "FAIL":
        return "HIGH"
    return "LOW"


def build_chunk_payload(items: List[Dict], has_more: bool, output_file: str) -> Dict:
    batch = {
        "total": len(items),
        "checked": len(items),
        "pass": sum(1 for x in items if x["status"] == "PASS"),
        "fail": sum(1 for x in items if x["status"] == "FAIL"),
        "review": sum(1 for x in items if x["status"] == "REVIEW"),
        "image_missing": sum(1 for x in items if x["status"] == "IMAGE_MISSING"),
    }

    failures_only = []
    for it in items:
        if it["status"] in ("FAIL", "REVIEW", "IMAGE_MISSING"):
            failures_only.append(
                {
                    "id": it["id"],
                    "severity": severity_for(it),
                    "codes": it["codes"],
                    "short_reason": it["short_reason"],
                }
            )

    return {
        "batch": batch,
        "results": items,
        "failures_only": failures_only,
        "next": "continue" if has_more else "done",
        "output_file": output_file.replace("\\", "/"),
    }


def build_summary_payload(chunk_files: List[str]) -> Dict:
    total = {"total": 0, "checked": 0, "pass": 0, "fail": 0, "review": 0, "image_missing": 0}
    failures_only: List[Dict] = []
    for fp in chunk_files:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        batch = data.get("batch", {})
        for k in total:
            total[k] += int(batch.get(k, 0))
        failures_only.extend(data.get("failures_only", []))

    return {
        "batch": total,
        "results": [],
        "failures_only": failures_only,
        "next": "done",
    }


def build_split_payload(items: List[Dict]) -> Dict[str, List[Dict]]:
    pass_items: List[Dict] = []
    fail_items: List[Dict] = []
    review_items: List[Dict] = []
    image_missing_items: List[Dict] = []

    for item in items:
        status = item.get("status")
        if status == "PASS":
            pass_items.append(item)
        elif status == "FAIL":
            fail_items.append(item)
        elif status == "REVIEW":
            review_items.append(item)
        elif status == "IMAGE_MISSING":
            image_missing_items.append(item)

    return {
        "pass": pass_items,
        "fail": fail_items,
        "review": review_items,
        "image_missing": image_missing_items,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full sweep DSL + image checker")
    parser.add_argument(
        "--json-files",
        nargs="+",
        default=DEFAULT_JSON_FILES,
        help="List of JSON dataset files to process.",
    )
    parser.add_argument(
        "--image-root",
        default=DEFAULT_IMAGE_ROOT,
        help="Image root directory used to resolve image_dir/id.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for review_chunk_XXX.json and summary. Defaults to first JSON file directory.",
    )
    parser.add_argument(
        "--review-dir",
        default=None,
        help="Directory to store split review files. Defaults to <output-dir>/Review.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Number of samples per chunk.",
    )
    parser.add_argument(
        "--summary-filename",
        default=DEFAULT_SUMMARY_FILENAME,
        help="Summary JSON filename written in output directory.",
    )
    parser.add_argument(
        "--pass-filename",
        default=DEFAULT_PASS_FILENAME,
        help="Output filename for PASS-only items.",
    )
    parser.add_argument(
        "--fail-filename",
        default=DEFAULT_FAIL_FILENAME,
        help="Output filename for FAIL-only items.",
    )
    parser.add_argument(
        "--review-filename",
        default=DEFAULT_REVIEW_FILENAME,
        help="Output filename for REVIEW-only items.",
    )
    parser.add_argument(
        "--image-missing-filename",
        default=DEFAULT_IMAGE_MISSING_FILENAME,
        help="Output filename for IMAGE_MISSING-only items.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable printing each chunk payload to stdout.",
    )
    parser.add_argument(
        "--write-chunks",
        action="store_true",
        help="Write review_chunk_XXX.json files. Disabled by default.",
    )
    parser.add_argument(
        "--write-summary",
        action="store_true",
        help="Write summary file (review_full_sweep_summary.json). Disabled by default.",
    )
    return parser.parse_args()


def to_rel_posix(path: str) -> str:
    return os.path.relpath(path, os.getcwd()).replace("\\", "/")


def main() -> None:
    args = parse_args()
    image_root = os.path.normpath(args.image_root)
    chunk_size = args.chunk_size if args.chunk_size > 0 else DEFAULT_CHUNK_SIZE
    output_dir = os.path.normpath(args.output_dir) if args.output_dir else os.path.dirname(args.json_files[0])

    json_files = [os.path.normpath(p) for p in args.json_files]
    default_json_mode = args.json_files == DEFAULT_JSON_FILES

    if default_json_mode:
        output_full = os.path.join(output_dir, "full.json")
        output_parts = sorted(glob.glob(os.path.join(output_dir, "full_part_*.json")))
        existing_defaults = [p for p in json_files if os.path.exists(p)]
        if os.path.exists(output_full):
            json_files = [os.path.normpath(output_full)]
        elif output_parts:
            json_files = [os.path.normpath(p) for p in output_parts]
        elif existing_defaults:
            json_files = existing_defaults

    os.makedirs(output_dir, exist_ok=True)
    if args.review_dir:
        review_dir = os.path.normpath(args.review_dir)
    elif os.path.basename(output_dir).lower() == "review":
        review_dir = output_dir
    else:
        review_dir = os.path.join(output_dir, "Review")
    os.makedirs(review_dir, exist_ok=True)

    all_samples: List[Tuple[str, Dict]] = []
    for fp in json_files:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        for s in data:
            all_samples.append((fp, s))

    total = len(all_samples)
    chunk_idx = 0
    start = 0
    all_results: List[Dict] = []
    chunk_paths: List[str] = []
    while start < total:
        end = min(start + chunk_size, total)
        chunk_idx += 1
        chunk_items = all_samples[start:end]
        results = [evaluate_sample(sample, src_file, image_root) for src_file, sample in chunk_items]
        all_results.extend(results)

        if args.write_chunks:
            out_name = f"review_chunk_{chunk_idx:03d}.json"
            out_path = os.path.join(output_dir, out_name)
            chunk_paths.append(out_path)
            payload = build_chunk_payload(results, has_more=(end < total), output_file=to_rel_posix(out_path))

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

            if not args.quiet:
                print(json.dumps(payload, ensure_ascii=False))
        start = end

    split_payload = build_split_payload(all_results)

    pass_path = os.path.join(review_dir, args.pass_filename)
    pass_json = {
        "total": len(split_payload["pass"]),
        "status": "PASS",
        "results": split_payload["pass"],
        "output_file": to_rel_posix(pass_path),
    }
    with open(pass_path, "w", encoding="utf-8") as f:
        json.dump(pass_json, f, ensure_ascii=False, indent=2)

    fail_path = os.path.join(review_dir, args.fail_filename)
    fail_json = {
        "total": len(split_payload["fail"]),
        "status": "FAIL",
        "results": split_payload["fail"],
        "output_file": to_rel_posix(fail_path),
    }
    with open(fail_path, "w", encoding="utf-8") as f:
        json.dump(fail_json, f, ensure_ascii=False, indent=2)

    review_path = os.path.join(review_dir, args.review_filename)
    review_json = {
        "total": len(split_payload["review"]),
        "status": "REVIEW",
        "results": split_payload["review"],
        "output_file": to_rel_posix(review_path),
    }
    with open(review_path, "w", encoding="utf-8") as f:
        json.dump(review_json, f, ensure_ascii=False, indent=2)

    image_missing_path = os.path.join(review_dir, args.image_missing_filename)
    image_missing_json = {
        "total": len(split_payload["image_missing"]),
        "status": "IMAGE_MISSING",
        "results": split_payload["image_missing"],
        "output_file": to_rel_posix(image_missing_path),
    }
    with open(image_missing_path, "w", encoding="utf-8") as f:
        json.dump(image_missing_json, f, ensure_ascii=False, indent=2)

    final_payload = {
        "batch": {
            "total": len(all_results),
            "checked": len(all_results),
            "pass": len(split_payload["pass"]),
            "fail": len(split_payload["fail"]),
            "review": len(split_payload["review"]),
            "image_missing": len(split_payload["image_missing"]),
        },
        "next": "done",
        "split_files": {
        "pass": to_rel_posix(pass_path),
        "fail": to_rel_posix(fail_path),
        "review": to_rel_posix(review_path),
        "image_missing": to_rel_posix(image_missing_path),
        },
    }

    if args.write_summary:
        summary = build_summary_payload(chunk_paths) if args.write_chunks else {
            "batch": final_payload["batch"],
            "results": [],
            "failures_only": [
                {
                    "id": it["id"],
                    "severity": severity_for(it),
                    "codes": it.get("codes", []),
                    "short_reason": it.get("short_reason", ""),
                }
                for it in all_results
                if it.get("status") in ("FAIL", "REVIEW", "IMAGE_MISSING")
            ],
            "next": "done",
        }
        summary_path = os.path.join(output_dir, args.summary_filename)
        summary["output_file"] = to_rel_posix(summary_path)
        summary["split_files"] = final_payload["split_files"]
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        final_payload["output_file"] = to_rel_posix(summary_path)

    print(json.dumps(final_payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
