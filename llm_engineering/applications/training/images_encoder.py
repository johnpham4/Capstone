"""Pre-encode images to tokens"""
import json
import torch
from PIL import Image
from pathlib import Path
from torchvision import transforms
from loguru import logger
from tqdm import tqdm
from models.modeling_geomagvit import GeoMAGVIT


def encode_dataset_images(input_json: str, output_json: str, images_dir: str, vq_model_path: str = "JO-KU/Geo-MAGVIT"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vq_model = GeoMAGVIT.from_pretrained(vq_model_path).to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    with open(input_json) as f:
        data = json.load(f)

    for sample in tqdm(data):
        img = Image.open(Path(images_dir) / sample["images"][0]).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            tokens = vq_model.get_code(img_tensor).squeeze(0).cpu()

        sample["image_tokens"] = tokens.tolist()

    with open(output_json, "w") as f:
        json.dump(data, f)

    logger.success(f"Encoded {len(data)} images")
