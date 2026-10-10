from __future__ import annotations

import torch


class CausalCacheCollator:
    def __init__(self, tokenizer, max_length, item_token_num):
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.item_token_num = int(item_token_num)

    def __call__(self, rows):
        prompts = [row["input_ids"] for row in rows]
        responses = [row["labels"] + self.tokenizer.eos_token for row in rows]
        prompt = self.tokenizer(
            prompts, return_tensors="pt", padding="longest", max_length=self.max_length,
            truncation=True, return_attention_mask=True,
        )
        full = self.tokenizer(
            text=responses, text_target=prompts, return_tensors="pt", padding="longest",
            max_length=self.max_length, truncation=True, return_attention_mask=True,
        )
        labels = full["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        labels[torch.where(full["labels"] != self.tokenizer.pad_token_id)] = -100
        ids_rows, pos_rows = [], []
        for row in labels:
            positions = torch.where(row != -100)[0]
            ids = row[positions]
            if ids.numel() < self.item_token_num:
                raise ValueError("Target response is shorter than item_token_num")
            ids_rows.append(ids[: self.item_token_num].long())
            pos_rows.append(
                torch.cat((positions[:1] - 1, positions[: self.item_token_num - 1])).long()
            )
        return {
            "input_ids": prompt["input_ids"],
            "attention_mask": prompt["attention_mask"],
            "full_input_ids": full["input_ids"],
            "full_attention_mask": full["attention_mask"],
            "target_ids": torch.stack(ids_rows),
            "teacher_positions": torch.stack(pos_rows),
        }
