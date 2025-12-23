import torch
from PIL import Image
from pathlib import Path
from torchvision import transforms
from loguru import logger


class T2DDataCollator:
    """Collator để encode image và format prompt"""

    def __init__(self, vq_model, prompter, image_transform=None):
        self.vq_model = vq_model
        self.prompter = prompter

        if image_transform is None:
            self.image_transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.image_transform = image_transform

    def __call__(self, batch):
        """Process batch: problem + image → tokens for training"""

        prompts = []
        image_tokens_list = []

        for sample in batch:
            problem = sample["problem"]
            image_file = sample["images"][0]
            images_dir = sample["images_dir"]

            image_path = Path(images_dir) / image_file

            image = Image.open(image_path).convert("RGB")
            image_tensor = self.image_transform(image).unsqueeze(0)

            with torch.no_grad():
                image_tokens = self.vq_model.get_code(image_tensor.to(self.vq_model.device))
                image_tokens = image_tokens.squeeze(0)

            image_tokens_list.append(image_tokens)
            prompts.append(problem)

        image_tokens_batch = torch.stack(image_tokens_list)

        # Process each prompt individually
        input_ids_list = []
        attention_masks_list = []

        for prompt in prompts:
            prompt_input_ids, prompt_attention_mask = self.prompter(prompt, "t2i_gen")
            input_ids_list.append(prompt_input_ids)
            attention_masks_list.append(prompt_attention_mask)

        # Pad all sequences to max length in batch
        max_length = max(ids.shape[1] for ids in input_ids_list)
        pad_token_id = self.prompter.text_tokenizer.pad_token_id

        padded_input_ids = []
        padded_attention_masks = []

        for input_ids, attention_mask in zip(input_ids_list, attention_masks_list):
            seq_len = input_ids.shape[1]
            if seq_len < max_length:
                # Pad to max_length
                padding = torch.full((1, max_length - seq_len), pad_token_id, dtype=input_ids.dtype)
                input_ids = torch.cat([input_ids, padding], dim=1)

                attention_padding = torch.zeros((1, max_length - seq_len), dtype=attention_mask.dtype)
                attention_mask = torch.cat([attention_mask, attention_padding], dim=1)

            padded_input_ids.append(input_ids)
            padded_attention_masks.append(attention_mask)

        input_ids = torch.cat(padded_input_ids, dim=0)
        attention_masks = torch.cat(padded_attention_masks, dim=0)

        labels = input_ids.clone()
        labels[labels == self.prompter.text_tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "labels": labels,
            "image_tokens": image_tokens_batch
        }
