from __future__ import annotations

import torch

from .data import extract_sid_tokens
from .prompt import format_prompt


def _find_last_subsequence(sequence, subsequence):
    width = len(subsequence)
    for start in range(len(sequence) - width, -1, -1):
        if sequence[start : start + width] == subsequence:
            return start
    return None


class MiniOneRecCollator:
    """Match MiniOneRec prompts and expose the three SID teacher positions."""

    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __call__(self, rows):
        prompts = [format_prompt(row["history_item_sid"], row["split"]) for row in rows]
        sid_tokens = [extract_sid_tokens(row["item_sid"]) for row in rows]
        responses = ["".join(tokens) + "\n" + self.tokenizer.eos_token for tokens in sid_tokens]
        full_texts = [prompt + response for prompt, response in zip(prompts, responses)]

        prompt_batch = self.tokenizer(
            prompts, return_tensors="pt", padding="longest", truncation=True,
            max_length=self.max_length, return_attention_mask=True,
        )
        full_batch = self.tokenizer(
            full_texts, return_tensors="pt", padding="longest", truncation=True,
            max_length=self.max_length + 3 + 8, return_attention_mask=True,
        )

        target_rows, teacher_rows = [], []
        for row_id, tokens in enumerate(sid_tokens):
            target_ids = [int(self.tokenizer.convert_tokens_to_ids(token)) for token in tokens]
            if any(token_id == self.tokenizer.unk_token_id for token_id in target_ids):
                raise ValueError(f"MiniOneRec SID contains an unknown tokenizer token: {tokens}")
            valid_length = int(full_batch["attention_mask"][row_id].sum().item())
            full_ids = full_batch["input_ids"][row_id, :valid_length].tolist()
            prompt_length = int(prompt_batch["attention_mask"][row_id].sum().item())
            prompt_ids = prompt_batch["input_ids"][row_id, :prompt_length].tolist()
            start = _find_last_subsequence(full_ids, target_ids)
            if start is None or start == 0:
                raise ValueError(f"Could not locate target SID tokens in encoded response: {tokens}")
            if full_ids[:start] != prompt_ids:
                raise ValueError("Prompt tokenization does not match the teacher prefix")
            target_rows.append(torch.tensor(target_ids, dtype=torch.long))
            teacher_rows.append(torch.arange(start - 1, start + len(target_ids) - 1))

        return {
            "input_ids": prompt_batch["input_ids"],
            "attention_mask": prompt_batch["attention_mask"],
            "full_input_ids": full_batch["input_ids"],
            "full_attention_mask": full_batch["attention_mask"],
            "target_ids": torch.stack(target_rows),
            "teacher_positions": torch.stack(teacher_rows),
        }
