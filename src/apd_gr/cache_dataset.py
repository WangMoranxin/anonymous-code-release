import json
import os
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset, Sampler

MANIFEST_NAME = "manifest.json"
PACKED_SHARD_FORMAT = "letter-dspark-packed-shard-v1"

def save_manifest(cache_dir: str, manifest: dict) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, MANIFEST_NAME)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    os.replace(tmp_path, path)

def load_manifest(cache_dir: str) -> dict:
    with open(os.path.join(cache_dir, MANIFEST_NAME), "r", encoding="utf-8") as f:
        return json.load(f)

def _dtype_from_name(name: str | torch.dtype | None) -> Optional[torch.dtype]:
    if name is None or isinstance(name, torch.dtype):
        return name
    normalized = str(name).lower().replace("torch.", "")
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"Unsupported cache dtype: {name}")

def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")

def _tensor_nbytes(value: torch.Tensor) -> int:
    return int(value.numel() * value.element_size())

class ShardedCacheWriter:
    def __init__(
        self,
        cache_dir: str,
        records_per_shard: int = 512,
        *,
        cache_dtype: str | torch.dtype | None = torch.bfloat16,
    ):
        self.cache_dir = cache_dir
        self.records_per_shard = int(records_per_shard)
        self.cache_dtype = _dtype_from_name(cache_dtype)
        self.current: List[Dict[str, torch.Tensor]] = []
        self.shards: List[str] = []
        self.shard_sizes: List[int] = []
        self.num_records = 0
        self.field_bytes: Dict[str, int] = {}
        self.field_dtypes: Dict[str, str] = {}
        os.makedirs(cache_dir, exist_ok=True)

    def _materialize(self, key: str, value: torch.Tensor) -> torch.Tensor:
        value = value.detach()
        if torch.is_floating_point(value) and self.cache_dtype is not None:
            value = value.to(device="cpu", dtype=self.cache_dtype)
        else:
            value = value.to(device="cpu")
        return value.contiguous()

    def _record_stats(self, record: Dict[str, torch.Tensor]) -> None:
        for key, value in record.items():
            self.field_bytes[key] = self.field_bytes.get(key, 0) + _tensor_nbytes(value)
            self.field_dtypes.setdefault(key, _dtype_name(value.dtype))

    def write(self, record: Dict[str, torch.Tensor]) -> None:
        materialized = {key: self._materialize(key, value) for key, value in record.items()}
        self._record_stats(materialized)
        self.current.append(materialized)
        self.num_records += 1
        if len(self.current) >= self.records_per_shard:
            self.flush()

    def _pack_current(self) -> Dict[str, torch.Tensor | str]:
        records = self.current
        lengths = [int(item["target_hidden_states"].shape[0]) for item in records]
        offsets = torch.zeros(len(records) + 1, dtype=torch.long)
        if lengths:
            offsets[1:] = torch.tensor(lengths, dtype=torch.long).cumsum(dim=0)
        context_dim = int(records[0]["target_hidden_states"].shape[1])
        if int(offsets[-1].item()) > 0:
            context_data = torch.cat([item["target_hidden_states"] for item in records], dim=0)
        else:
            context_data = torch.empty(
                (0, context_dim),
                dtype=records[0]["target_hidden_states"].dtype,
            )
        shard: Dict[str, torch.Tensor | str] = {
            "format": PACKED_SHARD_FORMAT,
            "target_ids": torch.stack([item["target_ids"].long() for item in records], dim=0),
            "target_hidden_states": context_data.contiguous(),
            "context_offsets": offsets,
        }
        if all("root_hidden" in item for item in records):
            shard["root_hidden"] = torch.stack(
                [item["root_hidden"] for item in records], dim=0
            )
        optional_keys = (
            "teacher_legal_logits",
        )
        for key in optional_keys:
            if all(key in item for item in records):
                if isinstance(records[0][key], torch.Tensor):
                    shard[key] = torch.stack([item[key] for item in records], dim=0)
        return shard

    def flush(self) -> None:
        if not self.current:
            return
        shard_name = f"shard_{len(self.shards):05d}.pt"
        shard_path = os.path.join(self.cache_dir, shard_name)
        torch.save(self._pack_current(), shard_path)
        self.shards.append(shard_name)
        self.shard_sizes.append(len(self.current))
        self.current = []

    def close(self) -> None:
        self.flush()

    def stats(self) -> dict:
        return {
            "field_bytes": dict(self.field_bytes),
            "field_dtypes": dict(self.field_dtypes),
            "cache_dtype": None if self.cache_dtype is None else _dtype_name(self.cache_dtype),
            "packed": True,
        }

class DraftCacheDataset(Dataset):
    """Reads packed target-hidden cache records for parallel_draft."""

    def __init__(
        self,
        cache_dir: str,
        records_per_shard: int = 512,
        *,
        use_mmap: bool = False,
        log_shard_loads: bool = False,
    ):
        self.cache_dir = cache_dir
        self.records_per_shard = int(records_per_shard)
        self.use_mmap = bool(use_mmap)
        self.log_shard_loads = bool(log_shard_loads)
        self.manifest = load_manifest(cache_dir)
        self.shards: List[str] = []
        self.shard_sizes: List[int] = []
        self.shard_start_indices: List[int] = []
        self.index: List[tuple[int, int]] = []
        self._loaded_shard_id = None
        self._loaded_records = None
        self.shard_load_count = 0
        self.context_is_projected = bool(self.manifest.get("context_is_projected", False))
        self.teacher_legal_sizes = [
            int(value) for value in self.manifest.get("teacher_legal_sizes", [])
        ]
        self._build_lazy_index()

    def _iter_rank_manifests(self):
        if not os.path.isdir(self.cache_dir):
            return
        for name in sorted(os.listdir(self.cache_dir)):
            rank_dir = os.path.join(self.cache_dir, name)
            path = os.path.join(rank_dir, MANIFEST_NAME)
            if name.startswith("rank_") and os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    yield json.load(f)

    def _sizes_from_count(self, num_records: int, shard_count: int) -> List[int]:
        sizes = []
        for shard_idx in range(shard_count):
            if shard_idx < shard_count - 1:
                shard_records = self.records_per_shard
            else:
                shard_records = int(num_records) - self.records_per_shard * (shard_count - 1)
            sizes.append(max(int(shard_records), 0))
        return sizes

    def _add_shards(
        self,
        shard_names: List[str],
        *,
        num_records: Optional[int] = None,
        shard_sizes: Optional[List[int]] = None,
    ) -> None:
        if shard_sizes is None:
            if num_records is None:
                raise ValueError("num_records is required when shard_sizes is absent")
            shard_sizes = self._sizes_from_count(int(num_records), len(shard_names))
        if len(shard_names) != len(shard_sizes):
            raise ValueError(
                f"Manifest shard count mismatch: {len(shard_names)} names vs {len(shard_sizes)} sizes"
            )
        for shard_name, shard_records in zip(shard_names, shard_sizes):
            shard_id = len(self.shards)
            self.shards.append(shard_name)
            self.shard_sizes.append(int(shard_records))
            self.shard_start_indices.append(len(self.index))
            for record_idx in range(max(int(shard_records), 0)):
                self.index.append((shard_id, record_idx))

    def _build_lazy_index(self):
        rank_manifests = list(self._iter_rank_manifests())
        if rank_manifests:
            for rank_manifest in rank_manifests:
                self._add_shards(
                    list(rank_manifest["shards"]),
                    num_records=int(rank_manifest.get("num_records", 0)),
                    shard_sizes=rank_manifest.get("shard_sizes"),
                )
            return

        self._add_shards(
            list(self.manifest["shards"]),
            num_records=int(self.manifest["num_records"]),
            shard_sizes=self.manifest.get("shard_sizes"),
        )

    def _torch_load_shard(self, shard_path: str):
        kwargs = {"map_location": "cpu"}
        if self.use_mmap:
            try:
                return torch.load(shard_path, mmap=True, weights_only=True, **kwargs)
            except TypeError:
                pass
            except RuntimeError:
                pass
            except ValueError:
                pass
        try:
            return torch.load(shard_path, weights_only=True, **kwargs)
        except TypeError:
            return torch.load(shard_path, **kwargs)

    def __len__(self):
        return len(self.index)

    def _load_shard(self, shard_id: int):
        if self._loaded_shard_id == shard_id:
            return self._loaded_records
        self._loaded_records = self._torch_load_shard(
            os.path.join(self.cache_dir, self.shards[shard_id])
        )
        self._loaded_shard_id = shard_id
        self.shard_load_count += 1
        if self.log_shard_loads:
            print(
                f"[cache] loaded shard {shard_id}/{len(self.shards)} "
                f"loads={self.shard_load_count} path={self.shards[shard_id]}",
                flush=True,
            )
        return self._loaded_records

    def _packed_item(self, shard: Dict[str, torch.Tensor], record_idx: int) -> Dict[str, torch.Tensor]:
        offsets = shard["context_offsets"]
        start = int(offsets[record_idx].item())
        end = int(offsets[record_idx + 1].item())
        record = {
            "target_ids": shard["target_ids"][record_idx],
            "target_hidden_states": shard["target_hidden_states"][start:end],
        }
        for key in ("teacher_legal_logits",):
            if key in shard:
                record[key] = shard[key][record_idx]
        if "root_hidden" in shard:
            record["root_hidden"] = shard["root_hidden"][record_idx]
        else:
            raise ValueError("Cache shard is missing root_hidden")
        if "teacher_legal_logits" in record:
            record["teacher_legal_sizes"] = torch.tensor(
                self.teacher_legal_sizes, dtype=torch.long
            )
        if self.context_is_projected:
            record["context_is_projected"] = torch.tensor(True)
        return record

    def __getitem__(self, idx: int):
        shard_id, record_idx = self.index[idx]
        shard = self._load_shard(shard_id)
        if isinstance(shard, list):
            raise ValueError(
                "Legacy list shard cache is no longer supported; regenerate with prepare_cache.py."
            )
        return self._packed_item(shard, record_idx)

class ShardLocalSampler(Sampler[int]):
    """Yields indices shard by shard, with optional shard and in-shard shuffling."""

    def __init__(
        self,
        dataset: DraftCacheDataset,
        *,
        shuffle_shards: bool = True,
        shuffle_within_shard: bool = True,
        seed: int = 0,
        num_replicas: int = 1,
        rank: int = 0,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.shuffle_shards = bool(shuffle_shards)
        self.shuffle_within_shard = bool(shuffle_within_shard)
        self.seed = int(seed)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.drop_last = bool(drop_last)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _shard_order(self) -> List[int]:
        shard_count = len(self.dataset.shards)
        if shard_count <= 0:
            return []
        if not self.shuffle_shards:
            return list(range(shard_count))
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        return torch.randperm(shard_count, generator=generator).tolist()

    def _rank_shards(self, rank: int) -> List[int]:
        return self._shard_order()[int(rank) :: self.num_replicas]

    def _rank_length(self, rank: int) -> int:
        return sum(self.dataset.shard_sizes[shard_id] for shard_id in self._rank_shards(rank))

    def _target_length(self) -> int:
        if self.num_replicas <= 1:
            return self._rank_length(self.rank)
        lengths = [self._rank_length(rank) for rank in range(self.num_replicas)]
        return min(lengths) if self.drop_last else max(lengths)

    def __iter__(self):
        indices: List[int] = []
        for shard_id in self._rank_shards(self.rank):
            start = self.dataset.shard_start_indices[shard_id]
            size = self.dataset.shard_sizes[shard_id]
            if self.shuffle_within_shard and size > 1:
                generator = torch.Generator()
                generator.manual_seed(self.seed + self.epoch * 1000003 + shard_id)
                order = torch.randperm(size, generator=generator).tolist()
            else:
                order = list(range(size))
            indices.extend(start + offset for offset in order)

        target_len = self._target_length()
        if self.num_replicas > 1:
            if self.drop_last:
                indices = indices[:target_len]
            elif len(indices) < target_len and indices:
                repeats = (target_len - len(indices) + len(indices) - 1) // len(indices)
                indices = indices + (indices * repeats)[: target_len - len(indices)]
        return iter(indices)

    def __len__(self):
        return self._target_length()

def _pad_2d(features: List[Dict], key: str):
    max_len = max(item[key].shape[0] for item in features)
    dim = features[0][key].shape[1]
    out = torch.zeros(
        (len(features), max_len, dim),
        dtype=features[0][key].dtype,
        device=features[0][key].device,
    )
    for idx, item in enumerate(features):
        out[idx, : item[key].shape[0]] = item[key]
    return out

class DraftCacheCollator:
    def __call__(self, features: List[Dict]):
        target_hidden_states = _pad_2d(features, "target_hidden_states")
        batch = {
            "target_hidden_states": target_hidden_states,
            "target_ids": torch.stack([item["target_ids"].long() for item in features], dim=0),
        }
        if "root_hidden" in features[0]:
            batch["root_hidden"] = torch.stack([item["root_hidden"] for item in features], dim=0)
        if "teacher_legal_logits" in features[0]:
            batch["teacher_legal_logits"] = torch.stack(
                [item["teacher_legal_logits"] for item in features], dim=0
            )
            batch["teacher_legal_sizes"] = torch.stack(
                [item["teacher_legal_sizes"].long() for item in features], dim=0
            )
        if "root_hidden" not in batch:
            raise ValueError("Cache record must contain root_hidden")

        batch["attention_mask"] = torch.zeros(
            target_hidden_states.shape[:2],
            dtype=torch.long,
            device=target_hidden_states.device,
        )
        for idx, item in enumerate(features):
            batch["attention_mask"][idx, : item["target_hidden_states"].shape[0]] = 1

        batch["input_ids"] = torch.zeros_like(batch["attention_mask"], dtype=torch.long)
        if any(bool(item.get("context_is_projected", False)) for item in features):
            batch["context_is_projected"] = True
        return batch
