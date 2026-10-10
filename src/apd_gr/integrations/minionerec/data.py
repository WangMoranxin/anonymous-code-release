from __future__ import annotations

import ast
import glob
import json
import os
import re

import pandas as pd
from torch.utils.data import Dataset


SID_RE = re.compile(r"(<a_\d+>)(<b_\d+>)(<c_\d+>)")


def extract_sid_tokens(value: str) -> list[str]:
    compact = str(value).replace(" ", "").strip()
    match = SID_RE.fullmatch(compact)
    if match is None:
        raise ValueError(f"Invalid MiniOneRec three-token SID: {value!r}")
    return list(match.groups())


def _work_root(data_path: str) -> str:
    path = os.path.abspath(data_path)
    if os.path.isdir(os.path.join(path, "data")):
        return path
    if os.path.basename(path) == "data" and os.path.isdir(path):
        return os.path.dirname(path)
    raise FileNotFoundError(
        f"MiniOneRec data_path must contain data/ and raw/ directories: {path}"
    )


def resolve_csv(data_path: str, dataset: str, mode: str) -> str:
    if mode not in {"train", "test"}:
        raise ValueError(f"Unsupported MiniOneRec split: {mode}")
    root = _work_root(data_path)
    matches = sorted(glob.glob(os.path.join(root, "data", mode, f"{dataset}_*.csv")))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one MiniOneRec {mode} CSV for {dataset}, found {matches}"
        )
    return matches[0]


def resolve_index(data_path: str, dataset: str) -> str:
    root = _work_root(data_path)
    candidates = [
        os.path.join(root, "raw", f"{dataset}.index.json"),
        os.path.join(root, "data", "index", f"{dataset}.index.json"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"MiniOneRec index not found; checked {candidates}")


def load_item_codes(data_path: str, dataset: str) -> dict[str, list[str]]:
    with open(resolve_index(data_path, dataset), encoding="utf-8") as handle:
        raw = json.load(handle)
    result = {str(item): extract_sid_tokens("".join(code)) for item, code in raw.items()}
    if not result:
        raise ValueError("MiniOneRec item-code index is empty")
    return result


class MiniOneRecDataset(Dataset):
    def __init__(self, data_path: str, dataset: str, mode: str):
        self.mode = mode
        self.path = resolve_csv(data_path, dataset, mode)
        frame = pd.read_csv(self.path, usecols=["history_item_sid", "item_sid"])
        self.rows = frame.to_dict("records")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = dict(self.rows[index])
        history = ast.literal_eval(str(row["history_item_sid"]))
        if not isinstance(history, list):
            raise ValueError(f"history_item_sid must be a list in {self.path}")
        row["history_item_sid"] = ["".join(extract_sid_tokens(value)) for value in history]
        row["item_sid"] = "".join(extract_sid_tokens(row["item_sid"]))
        row["split"] = self.mode
        return row
