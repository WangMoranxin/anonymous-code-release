"""Shared utilities for bundled backbone integrations."""
import argparse
import importlib.util
import json
import os
import random
import sys
from types import SimpleNamespace

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Prefer deterministic kernels while allowing third-party kernels without
    # deterministic implementations to warn instead of aborting a long run.
    torch.use_deterministic_algorithms(True, warn_only=True)


def parse_int_list(value):
    return [int(x) for x in value.split(",") if x] if isinstance(value, str) else [int(x) for x in value]


def add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_ckpt", required=True)
    parser.add_argument("--target_type", choices=["auto", "causal", "minionerec", "t5", "cofirec", "diger", "resid"], default="auto")
    parser.add_argument("--target_config", default="", help="DIGER YAML config (required for .pt checkpoints)")
    parser.add_argument("--feature_mapping", default="", help="ReSID GAOQ directory or item-code Parquet path")
    parser.add_argument("--repo_root", default="", help=argparse.SUPPRESS)
    parser.add_argument("--base_model", default="", help="Optional causal base-model override")
    parser.add_argument("--tokenizer_path", default="")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--dataset", default="Instruments")
    parser.add_argument("--index_file", default=".index.json")
    parser.add_argument("--max_his_len", type=int, default=20)
    parser.add_argument("--add_prefix", action="store_true")
    parser.add_argument("--his_sep", default=", ")
    parser.add_argument("--model_max_length", type=int, default=512)
    parser.add_argument("--item_token_num", type=int, default=4)
    parser.add_argument("--target_layer_ids", default="auto")
    return parser


def build_seqrec_args(args):
    values = vars(args).copy()
    defaults = dict(tasks="seqrec", train_prompt_sample_num="1", train_data_sample_num="-1",
        valid_prompt_id=0, sample_valid=True, valid_prompt_sample_num=2,
        only_train_response=False, sample_num=-1, test_task="SeqRec")
    for key, value in defaults.items(): values.setdefault(key, value)
    return SimpleNamespace(**values)


def load_repo_module(repo_root, filename):
    if not repo_root:
        raise ValueError("--repo_root is required to load the target repository dataset")
    path = os.path.join(os.path.abspath(repo_root), filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    name = f"_unified_repo_{os.path.splitext(filename)[0]}_{abs(hash(path))}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.abspath(repo_root))
    try: spec.loader.exec_module(module)
    finally: sys.path.pop(0)
    return module


def load_seqrec_dataset_class(repo_root):
    return load_repo_module(repo_root, "data.py").SeqRecDataset


def load_item_indices(data_path, dataset, index_file):
    with open(os.path.join(data_path, dataset, dataset + index_file), encoding="utf-8") as handle:
        return json.load(handle)


def semantic_token_id(tokenizer, token):
    ids = tokenizer(token, add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        raise ValueError(f"Semantic token {token!r} maps to {ids}, expected one id")
    return int(ids[0])


def build_position_token_ids(indices, tokenizer, item_token_num):
    positions = [set() for _ in range(item_token_num)]
    for code in indices.values():
        if len(code) != item_token_num: raise ValueError(f"Expected {item_token_num} item tokens, got {len(code)}")
        for pos, token in enumerate(code): positions[pos].add(semantic_token_id(tokenizer, token))
    result = [sorted(x) for x in positions]
    if any(not x for x in result): raise ValueError("A semantic position has no legal tokens")
    return result
