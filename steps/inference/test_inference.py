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

    logger.info("Loading models for inference test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use base Qwen model if no trained model path provided
    base_model = "Qwen/Qwen2.5-3B-Instruct"
    actual_model_path = model_path if model_path and Path(model_path).exists() else base_model

    logger.info(f"Loading model from: {actual_model_path}")
    if actual_model_path == base_model:
        logger.warning("No trained model found, using base Qwen model (not fine-tuned)")

    model = AutoModelForCausalLM.from_pretrained(
        actual_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(actual_model_path, trust_remote_code=True)

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

    input_ids, attention_mask = prompter(prompt, "t2i_gen")
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    logger.info("Generating image tokens")
    with torch.no_grad():
        # Use standard generate, model will output tokens including image part
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1024,  # 32x32 = 1024 tokens for image
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=int(prompter.sptids_dict['<|eoi|>']),
            temperature=1.0,
            do_sample=True,
            top_p=0.9
        )

    # Extract image tokens (after <soi>, before <eoi>)
    # Assuming tokens between soi_id and eoi_id are image tokens
    soi_id = int(prompter.sptids_dict['<|soi|>'])
    eoi_id = int(prompter.sptids_dict['<|eoi|>'])

    output_tokens = output[0].cpu().tolist()
    try:
        soi_idx = output_tokens.index(soi_id)
        eoi_idx = output_tokens.index(eoi_id, soi_idx)
        image_token_ids = output_tokens[soi_idx+1:eoi_idx]

        # Reshape to 32x32 grid
        image_tokens = torch.tensor(image_token_ids).reshape(1, 32, 32).to(device)
    except (ValueError, RuntimeError) as e:
        logger.error(f"Failed to extract image tokens: {e}")
        # Fallback: use first 1024 generated tokens
        generated_part = output[0][input_ids.shape[1]:]
        image_tokens = generated_part[:1024].reshape(1, 32, 32).to(device)

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
