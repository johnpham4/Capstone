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

        input_ids, attention_masks = self.prompter(prompts, "t2i_gen")

        labels = input_ids.clone()
        labels[labels == self.prompter.text_tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "labels": labels,
            "image_tokens": image_tokens_batch
        }
