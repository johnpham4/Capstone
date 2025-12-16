# """Data collator for GeoUni Text-to-Diagram training"""

# import torch
# from PIL import Image
# from torchvision import transforms
# from loguru import logger


# class T2DDataCollator:
#     """Collator để encode image và format prompt theo paper GeoUni"""

#     def __init__(self, vq_model, prompter, image_transform=None):
#         """
#         Args:
#             vq_model: Geo-MAGVIT model (frozen) để encode image → tokens
#             prompter: UniversalPrompting instance
#             image_transform: Transform cho image
#         """
#         self.vq_model = vq_model
#         self.prompter = prompter

#         if image_transform is None:
#             self.image_transform = transforms.Compose([
#                 transforms.Resize((256, 256)),
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.5], [0.5])
#             ])
#         else:
#             self.image_transform = image_transform

#     def __call__(self, batch):
#         """
#         Process batch: text + image → input_ids + labels

#         Input batch: List[{"text": str, "image": str}]
#         Output: {"input_ids": tensor, "attention_mask": tensor, "labels": tensor}
#         """

#         text_ids = []
#         image_ids = []

#         for sample in batch:
#             # 1. Tokenize text
#             text_tokens = self.prompter.text_tokenizer(
#                 sample["text"],
#                 add_special_tokens=False
#             )["input_ids"]
#             text_ids.append(text_tokens)

#             # 2. Encode image → tokens (256 tokens)
#             image = Image.open(sample["image"]).convert("RGB")
#             image_tensor = self.image_transform(image).unsqueeze(0)

#             with torch.no_grad():
#                 # VQ-VAE encode: image → discrete tokens
#                 image_tokens = self.vq_model.encode(image_tensor.to(self.vq_model.device))
#                 image_tokens = image_tokens.flatten()  # [16, 16] → [256]

#             image_ids.append(image_tokens)

#         # 3. Stack image tokens
#         image_ids = torch.stack(image_ids)

#         # 4. Format prompt theo paper
#         # Format: <bos> [system] <User> <t2i> [text] <Assistant> <soi> [image_tokens] <eoi> <eos>
#         input_ids, attention_mask, labels = self.prompter.t2i_prompt(text_ids, image_ids)

#         return {
#             "input_ids": input_ids,
#             "attention_mask": attention_mask,
#             "labels": labels
#         }
