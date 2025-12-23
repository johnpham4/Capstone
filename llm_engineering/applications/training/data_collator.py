import torch
from torch.nn.utils.rnn import pad_sequence


class T2DDataCollatorCached:
    def __init__(self, prompter):
        self.prompter = prompter
        self.pad_id = prompter.text_tokenizer.pad_token_id

    def __call__(self, batch):
        input_ids, masks, image_tokens = [], [], []

        for s in batch:
            ids, m = self.prompter(s["problem"], "t2i_gen")
            input_ids.append(ids.squeeze(0))
            masks.append(m.squeeze(0))
            image_tokens.append(torch.tensor(s["image_tokens"]))

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_id
        )
        masks = torch.nn.utils.rnn.pad_sequence(
            masks, batch_first=True, padding_value=0
        )

        labels = input_ids.clone()
        labels[labels == self.pad_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": masks,
            "labels": labels,
            "image_tokens": torch.stack(image_tokens)
        }

