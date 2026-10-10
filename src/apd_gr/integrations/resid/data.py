"""Native ReSID Parquet datasets and integer semantic-code collation."""
from __future__ import annotations

import os

import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset


def load_item_codes(mapping_path, feature_names=None):
    table = pq.read_table(mapping_path)
    names = list(feature_names or [name for name in table.column_names if name != "item_id"])
    if not names or "item_id" not in table.column_names:
        raise ValueError(f"Invalid ReSID item-code mapping schema: {table.schema}")
    missing = [name for name in names if name not in table.column_names]
    if missing:
        raise ValueError(f"Missing ReSID codebook columns: {missing}")
    rows = table.select(["item_id", *names]).to_pydict()
    sizes = [max(map(int, rows[name])) for name in names]
    offsets, running = [], 0
    for size in sizes:
        offsets.append(running)
        running += int(size)
    codes = {}
    for row, item_id in enumerate(rows["item_id"]):
        item_id = int(item_id)
        if item_id <= 0 or item_id in codes:
            raise ValueError(f"Invalid or duplicate ReSID item_id: {item_id}")
        code = []
        for name, offset, size in zip(names, offsets, sizes):
            value = int(rows[name][row])
            if not 1 <= value <= int(size):
                raise ValueError(f"Invalid {name}={value} for item {item_id}")
            code.append(value + offset)
        codes[item_id] = code
    expected = set(range(1, max(codes) + 1))
    if set(codes) != expected:
        raise ValueError("ReSID item IDs must be contiguous and start at 1")
    return codes, sizes


class ReSIDParquetDataset(Dataset):
    def __init__(self, data_path, split):
        path = os.path.join(os.path.abspath(data_path), split)
        table = pq.read_table(path, columns=["history", "target"])
        self.data = table.to_pydict()

    def __len__(self):
        return len(self.data["target"])

    def __getitem__(self, index):
        return self.data["history"][index], int(self.data["target"][index])


class ReSIDCollator:
    def __init__(self, item_codes, max_his_len):
        self.code_length = len(next(iter(item_codes.values())))
        self.max_his_len = int(max_his_len)
        self.item_codes = torch.zeros((max(item_codes) + 1, self.code_length), dtype=torch.long)
        for item_id, code in item_codes.items():
            self.item_codes[int(item_id)] = torch.as_tensor(code, dtype=torch.long)

    def __call__(self, rows):
        histories, targets = zip(*rows)
        histories = [list(map(int, row))[-self.max_his_len:] for row in histories]
        max_items = max(map(len, histories))
        item_ids = torch.zeros((len(rows), max_items), dtype=torch.long)
        item_mask = torch.zeros_like(item_ids, dtype=torch.bool)
        for row_id, history in enumerate(histories):
            if history:
                item_ids[row_id, :len(history)] = torch.as_tensor(history)
                item_mask[row_id, :len(history)] = True
        if int(item_ids.max()) >= self.item_codes.shape[0]:
            raise ValueError("Interaction contains an item absent from the ReSID mapping")
        input_ids = self.item_codes[item_ids].reshape(len(rows), -1)
        attention_mask = (
            item_mask.unsqueeze(-1)
            .expand(-1, -1, self.code_length)
            .reshape(len(rows), -1)
            .clone()
        )
        target_items = torch.as_tensor(targets, dtype=torch.long)
        if int(target_items.max()) >= self.item_codes.shape[0]:
            raise ValueError("Target contains an item absent from the ReSID mapping")
        target_ids = self.item_codes[target_items]
        return {"input_ids": input_ids, "attention_mask": attention_mask,
                "labels": target_ids.clone(), "target_ids": target_ids}
