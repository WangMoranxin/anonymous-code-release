from __future__ import annotations

import torch


class T5CacheCollator:
    def __init__(self, tokenizer, max_length, item_token_num):
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.item_token_num = int(item_token_num)

    def __call__(self, rows):
        inputs = self.tokenizer(
            [row["input_ids"] for row in rows], return_tensors="pt", padding="longest",
            truncation=True, max_length=self.max_length, return_attention_mask=True,
        )
        targets = self.tokenizer(
            [row["labels"] for row in rows], return_tensors="pt", padding="longest",
            add_special_tokens=True,
        )["input_ids"]
        target_ids = []
        for row in targets:
            ids = row[row != self.tokenizer.pad_token_id]
            if ids.numel() < self.item_token_num:
                raise ValueError("Target response is shorter than item_token_num")
            target_ids.append(ids[: self.item_token_num].long())
        labels = targets.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {**inputs, "labels": labels, "target_ids": torch.stack(target_ids)}
