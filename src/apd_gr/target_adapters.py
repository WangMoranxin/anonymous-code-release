"""Compose target execution backends with recommendation integrations."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import torch
import yaml
from transformers import AutoConfig, T5Config

from apd_gr.backends import (CausalBackend, CoFiRecBackend, DigerBackend, ReSIDBackend,
                      T5Backend)
from apd_gr.integrations.common import IntegerCodeTokenizer
from apd_gr.integrations.causal.cache_collator import CausalCacheCollator
from apd_gr.integrations.t5.cache_collator import T5CacheCollator
from apd_gr.integrations.loader import load_diger_module, load_integration_module


@dataclass(frozen=True)
class DraftProfile:
    layers: int
    hidden_size: int
    intermediate_size: int
    attention_heads: int


PROFILES = {
    "causal": DraftProfile(0, 0, 0, 0),  # zero means derive/inherit from target
    "minionerec": DraftProfile(0, 0, 0, 0),
    "resid": DraftProfile(0, 128, 512, 4),
    "t5": DraftProfile(0, 128, 512, 4),
    "cofirec": DraftProfile(0, 128, 512, 4),
    "diger": DraftProfile(0, 128, 512, 4),
}


def proportional_target_layer_ids(num_layers: int, start: int = 1, stride: int = 3):
    """LC-Rec-compatible rule: 16 layers -> [1, 4, 7, 10, 13]."""
    num_layers = int(num_layers)
    layers = list(range(int(start), num_layers, int(stride)))
    if not layers:
        layers = [max(0, num_layers - 1)]
    return layers


def resolve_target_layer_ids(value, num_layers: int):
    if value is None or str(value).strip().lower() in {"", "auto"}:
        return proportional_target_layer_ids(num_layers)
    return [int(x) for x in str(value).split(",") if x.strip()]


def _read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def detect_target_type(target_ckpt: str, override: str = "auto") -> str:
    """Detect causal PEFT, causal, vanilla T5, or CoFiRec from checkpoint metadata."""
    if override != "auto":
        return override
    target_ckpt = os.path.abspath(target_ckpt)
    if os.path.isfile(target_ckpt) and target_ckpt.endswith(".pt"):
        return "diger"
    if os.path.isfile(os.path.join(target_ckpt, "adapter_config.json")):
        return "causal"
    config = AutoConfig.from_pretrained(target_ckpt, local_files_only=True)
    architectures = {str(x).lower() for x in (getattr(config, "architectures", None) or [])}
    if "cofirec" in architectures or str(getattr(config, "model_type", "")).lower() == "cofirec":
        return "cofirec"
    if bool(getattr(config, "is_encoder_decoder", False)) or str(config.model_type).lower() == "t5":
        return "t5"
    return "causal"



class AdapterProfileMixin:
    @property
    def profile(self):
        base = PROFILES[self.kind]
        return DraftProfile(
            base.layers or len(proportional_target_layer_ids(self.num_layers)),
            base.hidden_size or self.hidden_size,
            base.intermediate_size
            or int(getattr(self.config, "intermediate_size", self.hidden_size * 4)),
            base.attention_heads
            or int(getattr(self.config, "num_attention_heads", 1)),
        )


class T5Adapter(AdapterProfileMixin, T5Backend):
    kind = "t5"

    def cache_collator(self, tokenizer):
        return T5CacheCollator(
            tokenizer, self.args.model_max_length, self.args.item_token_num
        )


class CoFiRecAdapter(AdapterProfileMixin, CoFiRecBackend):
    kind = "cofirec"

    def cache_collator(self, tokenizer):
        return T5CacheCollator(
            tokenizer, self.args.model_max_length, self.args.item_token_num
        )


class DigerAdapter(AdapterProfileMixin, DigerBackend):
    """DIGER's position-specific-code T5 wrapper."""
    kind = "diger"
    is_encoder_decoder = True
    uses_integer_codes = True

    @property
    def required_item_token_num(self):
        return int(self.diger_config["code_length"])

    def __init__(self, args):
        self.args = args
        self.target_ckpt = os.path.abspath(args.target_ckpt)
        if not args.repo_root:
            raise ValueError("DIGER requires --repo_root")
        if not args.target_config:
            raise ValueError("DIGER requires --target_config")
        with open(os.path.abspath(args.target_config), encoding="utf-8") as handle:
            self.diger_config = yaml.safe_load(handle)
        data_path = args.data_path or self.diger_config.get("data_path", "./dataset")
        if not os.path.isabs(data_path):
            data_path = os.path.abspath(os.path.join(self.args.repo_root, data_path))
        self.diger_config["data_path"] = os.path.abspath(data_path)
        cfg = self.diger_config
        self.config = T5Config(
            num_layers=cfg["encoder_layers"], num_decoder_layers=cfg["decoder_layers"],
            d_model=cfg["d_model"], d_ff=cfg["d_ff"], num_heads=cfg["num_heads"],
            d_kv=cfg["d_kv"], dropout_rate=cfg["dropout_rate"],
            activation_function=cfg["activation_function"], vocab_size=1,
            pad_token_id=0, eos_token_id=300, decoder_start_token_id=0,
            feed_forward_proj=cfg["feed_forward_proj"], n_positions=cfg["max_length"])
        self.code_path = os.path.splitext(self.target_ckpt)[0] + ".code.json"
        if not os.path.isfile(self.code_path):
            raise FileNotFoundError(f"DIGER item-code file not found: {self.code_path}")
        self.item_codes = _read_json(self.code_path)

    def load_tokenizer(self, padding_side="right"):
        return IntegerCodeTokenizer(self.diger_config["code_num"])

    def cache_collator(self, tokenizer):
        from apd_gr.integrations.diger.collator import DigerCacheCollator
        return DigerCacheCollator(self.item_codes, self.diger_config["code_length"])

    def cache_dataset(self):
        module = load_diger_module(self.args.repo_root, "data.py", "diger_data")
        _, num_items, train, _, _ = module.load_split_data(self.diger_config)
        return module.SequentialSplitDataset(self.diger_config, num_items, train)

    def test_dataset(self):
        module = load_diger_module(self.args.repo_root, "data.py", "diger_data")
        _, num_items, _, _, test = module.load_split_data(self.diger_config)
        return module.SequentialSplitDataset(self.diger_config, num_items, test)

    def position_token_ids(self):
        return [list(range(int(self.diger_config["code_num"])))
                for _ in range(int(self.diger_config["code_length"]))]

    def item_code_dict(self):
        return {str(idx): list(map(int, code)) for idx, code in enumerate(self.item_codes[1:], 1)}


class ReSIDAdapter(AdapterProfileMixin, ReSIDBackend):
    """ReSID T5 with a raw state_dict and Parquet integer-code data."""
    kind = "resid"
    uses_integer_codes = True

    def __init__(self, args):
        self.args, self.target_ckpt = args, os.path.abspath(args.target_ckpt)
        if not os.path.isfile(self.target_ckpt):
            raise FileNotFoundError(f"ReSID checkpoint not found: {self.target_ckpt}")
        if not args.target_config or not args.feature_mapping:
            raise ValueError("ReSID requires --target_config and --feature_mapping")
        with open(os.path.abspath(args.target_config), encoding="utf-8") as handle:
            self.resid_config = yaml.safe_load(handle)
        cfg = self.resid_config
        self.data_path = os.path.abspath(args.data_path)
        mapping = os.path.abspath(args.feature_mapping)
        if os.path.isdir(mapping):
            direct = os.path.join(mapping, "item_code_mapping.parquet")
            nested = os.path.join(mapping, "item_feature", "item_code_mapping.parquet")
            mapping = direct if os.path.isfile(direct) else nested
        if not os.path.isfile(mapping):
            raise FileNotFoundError(f"ReSID item-code mapping not found: {mapping}")
        self.mapping_path = mapping
        self.item_codes, self.codebook_sizes = self._data_module().load_item_codes(
            mapping, cfg.get("feature"))
        code_length = len(self.codebook_sizes)
        vocab_size = max(max(code) for code in self.item_codes.values()) + 2
        self.config = T5Config(
            num_layers=int(cfg["encoder_layers"]), num_decoder_layers=int(cfg["decoder_layers"]),
            d_model=int(cfg["d_model"]), d_ff=int(cfg["d_ff"]),
            num_heads=int(cfg["num_heads"]), d_kv=int(cfg["d_kv"]),
            dropout_rate=float(cfg.get("dropout_rate", 0.1)), vocab_size=vocab_size,
            pad_token_id=0, eos_token_id=vocab_size - 1, decoder_start_token_id=0,
            feed_forward_proj=cfg.get("feed_forward_proj", "relu"),
            layer_norm_epsilon=float(cfg.get("layer_norm_epsilon", 1e-8)),
            initializer_factor=float(cfg.get("initializer_factor", 0.02)),
            n_positions=int(cfg.get("max_len", args.max_his_len)) * code_length)

    @property
    def required_item_token_num(self):
        return len(self.codebook_sizes)

    def _data_module(self):
        root = os.path.join(os.path.dirname(__file__), "integrations", "resid")
        return load_integration_module(root, "data.py", "resid_data")

    def load_tokenizer(self, padding_side="right"):
        return IntegerCodeTokenizer(self.config.vocab_size, 0, self.config.eos_token_id,
                                    "resid_integer_code")

    def cache_collator(self, tokenizer):
        return self._data_module().ReSIDCollator(self.item_codes, self.args.max_his_len)

    def cache_dataset(self):
        return self._data_module().ReSIDParquetDataset(self.data_path, "train")

    def test_dataset(self):
        return self._data_module().ReSIDParquetDataset(self.data_path, "test")

    def position_token_ids(self):
        offset, positions = 0, []
        for size in self.codebook_sizes:
            positions.append(list(range(offset + 1, offset + int(size) + 1)))
            offset += int(size)
        return positions

    def item_code_dict(self):
        return {str(key): list(value) for key, value in self.item_codes.items()}


class CausalAdapter(AdapterProfileMixin, CausalBackend):
    kind = "causal"

    def cache_collator(self, tokenizer):
        return CausalCacheCollator(
            tokenizer, self.args.model_max_length, self.args.item_token_num
        )


class MiniOneRecAdapter(CausalAdapter):
    """MiniOneRec Qwen target with CSV histories and three-token SIDs."""

    kind = "minionerec"
    uses_integer_codes = True

    def __init__(self, args):
        super().__init__(args)
        from apd_gr.integrations.minionerec.data import load_item_codes
        self.data_path = os.path.abspath(args.data_path)
        self.item_codes = load_item_codes(self.data_path, args.dataset)
        self._tokenizer = None

    @property
    def required_item_token_num(self):
        return 3

    def load_tokenizer(self, padding_side="right"):
        tokenizer = super().load_tokenizer(padding_side=padding_side)
        tokenizer.truncation_side = "left"
        self._tokenizer = tokenizer
        return tokenizer

    def cache_collator(self, tokenizer):
        from apd_gr.integrations.minionerec.collator import MiniOneRecCollator
        active_tokenizer = tokenizer or self._tokenizer
        if active_tokenizer is None:
            raise RuntimeError("Load the MiniOneRec tokenizer before creating its collator")
        return MiniOneRecCollator(active_tokenizer, self.args.model_max_length)

    def cache_dataset(self):
        from apd_gr.integrations.minionerec.data import MiniOneRecDataset
        return MiniOneRecDataset(self.data_path, self.args.dataset, "train")

    def test_dataset(self):
        from apd_gr.integrations.minionerec.data import MiniOneRecDataset
        return MiniOneRecDataset(self.data_path, self.args.dataset, "test")

    def position_token_ids(self):
        if self._tokenizer is None:
            raise RuntimeError("Load the MiniOneRec tokenizer before resolving legal SID tokens")
        positions = [set() for _ in range(self.required_item_token_num)]
        for code in self.item_codes.values():
            for pos, token in enumerate(code):
                token_id = int(self._tokenizer.convert_tokens_to_ids(token))
                if token_id == self._tokenizer.unk_token_id:
                    raise ValueError(f"Unknown MiniOneRec SID token: {token}")
                positions[pos].add(token_id)
        return [sorted(values) for values in positions]

    def item_code_dict(self):
        return self.item_codes


def apply_adapter_item_token_num(args, adapter):
    required = adapter.required_item_token_num
    if required is not None:
        args.item_token_num = int(required)
    return int(args.item_token_num)


def make_target_adapter(args):
    kind = detect_target_type(args.target_ckpt, args.target_type)
    integration = {
        "causal": "causal",
        "minionerec": "minionerec",
        "t5": "t5",
        "cofirec": "t5",
        "diger": "diger",
        "resid": "resid",
    }[kind]
    bundled_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "integrations", integration)
    if not os.path.isdir(bundled_root):
        raise FileNotFoundError(f"Bundled integration is missing: {bundled_root}")
    # Force all executable integration code to the bundled copy. External paths
    # provide checkpoints, configuration and data only.
    args.repo_root = bundled_root
    cls = {"causal": CausalAdapter, "minionerec": MiniOneRecAdapter,
           "t5": T5Adapter, "cofirec": CoFiRecAdapter,
           "diger": DigerAdapter, "resid": ReSIDAdapter}[kind]
    return cls(args)
