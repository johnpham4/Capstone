import gradio as gr
import json
import os

JSON_PATH = "dataset/data/splits/Minh/diagrams.json"
IMAGE_FOLDER = "dataset/data/splits/Minh/images"
OUTPUT_BAD = "bad_images.json"

# ===== LOAD DATA =====
with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# load bad ids
if os.path.exists(OUTPUT_BAD):
    with open(OUTPUT_BAD, "r") as f:
        bad_ids = set(json.load(f))   # dùng set để lookup nhanh
else:
    bad_ids = set()


# ===== HELPER =====
def find_next_valid_index(i):
    """skip những id đã dislike"""
    while i < len(data) and data[i]["id"] in bad_ids:
        i += 1
    return i


def get_item(i):

    i = find_next_valid_index(i)

    if i >= len(data):
        return "DONE", None, i

    item = data[i]

    id_ = item["id"]
    caption_vn = item.get("caption_vn", "")

    img_path = os.path.join(IMAGE_FOLDER, f"img_{id_}.png")

    text = f"""
### ID: {id_}

**Caption VN**

{caption_vn}
"""

    return text, img_path, i


def like(i):
    i += 1
    return (*get_item(i),)


def dislike(i):
    item = data[i]
    id_ = item["id"]

    bad_ids.add(id_)

    with open(OUTPUT_BAD, "w") as f:
        json.dump(list(bad_ids), f, indent=2)

    i += 1
    return (*get_item(i),)


# ===== INIT =====
init_text, init_img, init_idx = get_item(0)


# ===== UI =====
with gr.Blocks() as demo:

    state = gr.State(init_idx)

    with gr.Row():

        with gr.Column(scale=1):
            text_box = gr.Markdown(init_text)

        with gr.Column(scale=1):
            image = gr.Image(init_img)

    with gr.Row():
        like_btn = gr.Button("👍 Like", variant="primary")
        dislike_btn = gr.Button("👎 Dislike", variant="stop")

    like_btn.click(
        like,
        inputs=state,
        outputs=[text_box, image, state],
    )

    dislike_btn.click(
        dislike,
        inputs=state,
        outputs=[text_box, image, state],
    )


demo.launch()