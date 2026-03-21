def tokenize_and_mask_response_only(tokenizer, samples, max_seq_length, response_marker="### Response:"):
    texts = samples["text"]
    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=max_seq_length,
        padding=False,
        add_special_tokens=True,
    )

    marker_ids = tokenizer(
        response_marker,
        add_special_tokens=False
    )["input_ids"]

    labels = []

    for input_ids in encodings["input_ids"]:
        start_idx = None

        for i in range(len(input_ids) - len(marker_ids) + 1):
            if input_ids[i : i + len(marker_ids)] == marker_ids:
                start_idx = i + len(marker_ids)
                break

        if start_idx is None:
            label = [-100] * len(input_ids)
        else:
            label = [-100] * start_idx + input_ids[start_idx:]

        label = label[: len(input_ids)]
        labels.append(label)

    encodings["labels"] = labels
    return encodings

