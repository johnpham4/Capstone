import argparse
import json
import os
import re
from datetime import datetime

import gradio as gr
from PIL import Image

DEFAULT_JSON_PATH = "dataset/Minh/full.json"
DEFAULT_IMAGE_FOLDER = "dataset/Minh/images"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gradio reviewer for geometry samples")
    parser.add_argument(
        "--json-path",
        default=os.getenv("REVIEW_JSON_PATH", DEFAULT_JSON_PATH),
        help="Path to review JSON file (supports list or {'results': [...]})",
    )
    parser.add_argument(
        "--image-folder",
        default=os.getenv("REVIEW_IMAGE_FOLDER", DEFAULT_IMAGE_FOLDER),
        help="Path to image folder",
    )
    parser.add_argument(
        "--output-bad",
        default=os.getenv("REVIEW_OUTPUT_BAD", ""),
        help="Path to output bad json (optional)",
    )
    return parser.parse_args()


def load_items(json_path: str) -> list[dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        return raw

    if isinstance(raw, dict):
        results = raw.get("results")
        if isinstance(results, list):
            return results

    raise ValueError(
        "Unsupported JSON format. Expected a list or an object containing 'results'."
    )


args = parse_args()
JSON_PATH = args.json_path
IMAGE_FOLDER = args.image_folder
OUTPUT_BAD = args.output_bad or f"bad_{os.path.splitext(os.path.basename(JSON_PATH))[0]}.json"


# ===== LOAD DATA =====
items = load_items(JSON_PATH)
TOTAL_ITEMS = len(items)


def _load_bad_data() -> tuple[set[str], dict[str, str]]:
    """Load bad image ids and optional reasons from json with backward compatibility."""
    if not os.path.exists(OUTPUT_BAD):
        return set(), {}

    with open(OUTPUT_BAD, "r", encoding="utf-8") as f:
        raw_bad = json.load(f)

    if not raw_bad:
        return set(), {}

    # Old format: [id1, id2, ...]
    if isinstance(raw_bad[0], (str, int)):
        bad_records = {str(item): "" for item in raw_bad}
        return set(bad_records.keys()), bad_records

    # New format: [{"id": "...", "reason": "..."}, ...]
    bad_records = {
        str(item.get("id")): (item.get("reason") or "")
        for item in raw_bad
        if isinstance(item, dict) and item.get("id") is not None
    }
    return set(bad_records.keys()), bad_records


bad_ids, bad_records = _load_bad_data()


# ===== HELPER =====
def _item_key(item: dict, fallback_index: int) -> str:
    """Return a stable key for each item, preferring id then image_dir."""
    if item.get("id") is not None:
        return str(item["id"])

    image_dir = item.get("image_dir")
    if isinstance(image_dir, str) and image_dir:
        match = re.search(r"img_(\d+)\.png$", image_dir)
        if match:
            return match.group(1)
        return os.path.splitext(os.path.basename(image_dir))[0]

    return str(fallback_index)


def _item_image_path(item: dict, key: str) -> str | None:
    """Resolve image path for both legacy and current json formats."""
    candidates: list[str] = []
    image_dir = item.get("image_dir")
    if isinstance(image_dir, str) and image_dir:
        if os.path.isabs(image_dir):
            candidates.append(image_dir)
        else:
            # Prefer path relative to dataset root where json lives
            dataset_root = os.path.dirname(JSON_PATH)
            candidates.append(os.path.join(dataset_root, image_dir))

            # Fallback to IMAGE_FOLDER + file name
            base_name = os.path.basename(image_dir)
            candidates.append(os.path.join(IMAGE_FOLDER, base_name))

            # Common mismatch: JSON uses img_*.png while folder has diagram_*.png
            if base_name.startswith("img_"):
                candidates.append(
                    os.path.join(IMAGE_FOLDER, base_name.replace("img_", "diagram_", 1))
                )

    # Generic fallback by numeric key
    candidates.append(os.path.join(IMAGE_FOLDER, f"img_{key}.png"))
    candidates.append(os.path.join(IMAGE_FOLDER, f"diagram_{key}.png"))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    return None


def find_next_valid_index(i: int) -> int:
    """Skip ids that were already disliked."""
    while i < len(items) and _item_key(items[i], i) in bad_ids:
        i += 1
    return i


def get_item(i: int):
    i = find_next_valid_index(i)

    if i >= len(items):
        done_text = f"DONE\n\nDa duyet: {TOTAL_ITEMS}/{TOTAL_ITEMS} cau"
        return done_text, None, i

    item = items[i]
    id_ = _item_key(item, i)
    problem_text = (
        item.get("problem")
        or item.get("caption_vn")
        or item.get("instruction")
        or ""
    )
    dsl_answer = item.get("answer") or item.get("dsl") or ""
    status = item.get("status") or ""
    codes = item.get("codes") or []
    short_reason = item.get("short_reason") or ""
    action = item.get("action") or ""
    codes_text = ", ".join(str(c) for c in codes) if isinstance(codes, list) else str(codes)
    img_path = _item_image_path(item, id_)
    img_value = None
    image_note = ""
    if img_path is None:
        image_note = "\n\n[WARNING] No image file found for this item."
    else:
        try:
            img_value = Image.open(img_path).copy()
        except Exception:
            image_note = "\n\n[WARNING] Image file exists but cannot be loaded."

    text = f"""
### Source JSON: {JSON_PATH}

### Progress: {i + 1}/{TOTAL_ITEMS} | Total Questions: {TOTAL_ITEMS}

### ID: {id_}

**Problem**

{problem_text}{image_note}

**DSL**

```text
{dsl_answer}
```
"""
    return text, img_value, i


def _save_bad_data() -> None:
    payload = [
        {
            "id": bad_id,
            "reason": bad_records.get(bad_id, ""),
            "updated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
        for bad_id in sorted(bad_ids)
    ]
    with open(OUTPUT_BAD, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def like(i: int):
    i += 1
    return (*get_item(i), "")


def dislike(i: int, reason: str):
    item = items[i]
    id_ = _item_key(item, i)
    reason = (reason or "").strip()

    bad_ids.add(id_)
    bad_records[id_] = reason
    _save_bad_data()

    i += 1
    return (*get_item(i), "")


# ===== INIT =====
init_text, init_img, init_idx = get_item(0)


# ===== UI =====
with gr.Blocks() as demo:
    state = gr.State(init_idx)

    with gr.Row():
        with gr.Column(scale=1):
            text_box = gr.Markdown(init_text)

        with gr.Column(scale=1):
            image = gr.Image(value=init_img, type="pil", show_label=False)

    with gr.Row():
        like_btn = gr.Button("👍 Like", variant="primary")
        dislike_btn = gr.Button("👎 Dislike", variant="stop")

    reason_box = gr.Textbox(
        label="Nguyen nhan (tuy chon)",
        placeholder="Nhap ly do dislike... (co the de trong)",
        lines=2,
    )

    like_btn.click(
        like,
        inputs=state,
        outputs=[text_box, image, state, reason_box],
    )

    dislike_btn.click(
        dislike,
        inputs=[state, reason_box],
        outputs=[text_box, image, state, reason_box],
    )

    # Enter in reason box triggers dislike immediately.
    # Empty reason is allowed.
    reason_box.submit(
        dislike,
        inputs=[state, reason_box],
        outputs=[text_box, image, state, reason_box],
    )


demo.launch()