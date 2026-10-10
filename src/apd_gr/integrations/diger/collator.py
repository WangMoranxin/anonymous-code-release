from __future__ import annotations

import torch


class DigerCacheCollator:
    def __init__(self, item_codes, code_length):
        self.item_codes = torch.as_tensor(item_codes, dtype=torch.long)
        self.code_length = int(code_length)

    def __call__(self, rows):
        histories, targets = zip(*rows)
        histories = [torch.as_tensor(row, dtype=torch.long) for row in histories]
        max_items = max(row.numel() for row in histories)
        item_ids = torch.zeros((len(rows), max_items), dtype=torch.long)
        item_mask = torch.zeros_like(item_ids, dtype=torch.bool)
        for row_id, row in enumerate(histories):
            item_ids[row_id, : row.numel()] = row
            item_mask[row_id, : row.numel()] = True
        codes = self.item_codes[item_ids].reshape(len(rows), -1)
        code_mask = item_mask.unsqueeze(-1).expand(
            -1, -1, self.code_length
        ).reshape(len(rows), -1).clone()
        codes = codes.masked_fill(~code_mask, -1)
        target_item_ids = torch.as_tensor(targets, dtype=torch.long).reshape(-1)
        target_ids = self.item_codes[target_item_ids]
        return {
            "input_ids": codes,
            "attention_mask": code_mask,
            "labels": target_ids,
            "target_ids": target_ids,
        }
