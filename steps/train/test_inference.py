from zenml import step
from PIL import Image
from pathlib import Path
import torch
import numpy as np
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer
from torchvision import transforms

from models.modeling_geomagvit import GeoMAGVIT
from models.prompting_utils import UniversalPrompting


@step
def test_inference_step(
    model_path: str,
    vq_model_path: str,
    test_image_path: str,
    output_dir: str
) -> str:
    """Test model inference: image -> LLM -> generate new image"""

    logger.info("Loading models for inference test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    vq_model = GeoMAGVIT.from_pretrained(vq_model_path).to(device).eval()

    prompter = UniversalPrompting(
        tokenizer,
        max_len=4096,
        special_tokens=("<|soi|>", "<|eoi|>", "<|t2i|>", "<|mmu|>", "<|mix|>",
                      "<formalization>", "</formalization>", "<answer>", "</answer>"),
        ignore_id=-100
    )

    logger.info(f"Loading test image: {test_image_path}")
    image = Image.open(test_image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        input_tokens = vq_model.get_code(image_tensor)

    prompt = "Vẽ đoạn thẳng CD có độ dài 8"
    logger.info(f"Test prompt: {prompt}")

    input_ids, _ = prompter(prompt, "t2i_gen")
    input_ids = input_ids.to(device)

    logger.info("Generating image tokens")
    with torch.no_grad():
        output_tokens = model.t2i_generate(
            input_ids=input_ids,
            max_new_tokens=1024,
            pad_token_id=tokenizer.pad_token_id,
            temperature=1.0
        )

    logger.info("Decoding tokens to image")
    with torch.no_grad():
        generated_image = vq_model.decode_code(output_tokens)

    generated_image = torch.clamp((generated_image + 1.0) / 2.0, 0.0, 1.0) * 255.0
    generated_image = generated_image[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_file = output_path / "test_generated.png"
    Image.fromarray(generated_image).save(output_file)

    logger.success(f"Test image generated: {output_file}")

    return str(output_file)
