import argparse
import json
import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from apd_gr.cache_dataset import ShardedCacheWriter, save_manifest
from apd_gr.modeling_parallel_draft import ParallelDraftModel
from apd_gr.target_adapters import (apply_adapter_item_token_num, make_target_adapter,
                             resolve_target_layer_ids)
from apd_gr.utils import (add_common_args, build_position_token_ids, build_seqrec_args,
                   load_item_indices, load_seqrec_dataset_class, parse_int_list, set_seed)


def init_dist():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1 and not dist.is_initialized():
        torch.cuda.set_device(local); dist.init_process_group("nccl")
    return world, rank, local


def parse_args():
    parser = add_common_args(argparse.ArgumentParser(description="Build a unified draft cache"))
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--local_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--records_per_shard", type=int, default=512)
    parser.add_argument("--sample_fraction", type=float, default=1.0)
    parser.add_argument("--sample_count", type=int)
    parser.add_argument("--sample_seed", type=int)
    parser.add_argument("--cache_dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--teacher_cache", choices=["root_only", "full_legal"], default="root_only")
    parser.add_argument("--project_context", action="store_true")
    parser.add_argument("--draft_ckpt_for_projection")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def sample_dataset(dataset, args, rank):
    total = len(dataset)
    count = min(total, args.sample_count) if args.sample_count else min(total, max(1, round(total * args.sample_fraction)))
    if count == total: return dataset, None
    indices = torch.randperm(total, generator=torch.Generator().manual_seed(args.sample_seed or args.seed))[:count].tolist()
    if rank == 0: print(f"Using cache sample {count}/{total}", flush=True)
    return Subset(dataset, indices), {"sample_count": count, "sample_total": total}


def load_projection(args, device, dtype):
    if not args.project_context: return None
    if not args.draft_ckpt_for_projection: raise ValueError("--project_context requires --draft_ckpt_for_projection")
    model = ParallelDraftModel.from_pretrained(args.draft_ckpt_for_projection, local_files_only=True)
    return model.to(device=device, dtype=dtype).eval()


def merge(manifests, key):
    result = {}
    for manifest in manifests:
        for name, value in manifest.get(key, {}).items():
            result[name] = result.get(name, 0) + value if key == "field_bytes" else value
    return result


@torch.no_grad()
def main():
    args = parse_args(); set_seed(args.seed)
    world, rank, local = init_dist()
    device = torch.device("cuda", local) if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    adapter = make_target_adapter(args)
    apply_adapter_item_token_num(args, adapter)
    layers = resolve_target_layer_ids(args.target_layer_ids, adapter.num_layers)
    args.target_layer_ids = ",".join(map(str, layers))
    if rank == 0:
        print(f"Auto architecture: target_layers={layers}, draft_layers={len(layers)}", flush=True)
    tokenizer = adapter.load_tokenizer()
    target = adapter.load_model(tokenizer, device, dtype)
    if max(layers) >= adapter.num_layers:
        raise ValueError(f"target_layer_ids={layers} exceeds {adapter.kind} layers={adapter.num_layers}")
    if adapter.uses_integer_codes:
        indices = adapter.item_code_dict()
        legal_ids = adapter.position_token_ids()
        dataset = adapter.cache_dataset()
    else:
        indices = load_item_indices(args.data_path, args.dataset, args.index_file)
        legal_ids = build_position_token_ids(indices, tokenizer, args.item_token_num)
        Dataset = load_seqrec_dataset_class(args.repo_root)
        dataset = Dataset(build_seqrec_args(args), mode="train", prompt_sample_num=1, sample_num=-1)
    dataset, sample_info = sample_dataset(dataset, args, rank)
    if world > 1: dataset = Subset(dataset, range(rank, len(dataset), world))
    loader = DataLoader(dataset, batch_size=args.local_batch_size, shuffle=False,
        collate_fn=adapter.cache_collator(tokenizer), num_workers=args.num_workers, pin_memory=True)
    projection = load_projection(args, device, dtype)
    rank_dir = os.path.join(args.cache_dir, f"rank_{rank:05d}")
    writer = ShardedCacheWriter(rank_dir, records_per_shard=args.records_per_shard,
                                cache_dtype=args.cache_dtype)
    for batch in tqdm(loader, disable=rank != 0, desc=f"prepare-{adapter.kind}-cache"):
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        context, root, teacher = adapter.extract_cache(target, batch, layers, legal_ids, args.teacher_cache)
        if projection is not None: context = projection.project_target_context(context)
        lengths = batch["attention_mask"].sum(1).long()
        for row, length in enumerate(lengths.tolist()):
            record = {"target_ids": batch["target_ids"][row], "target_hidden_states": context[row, :length],
                      "root_hidden": root[row]}
            if teacher is not None: record["teacher_legal_logits"] = teacher[row]
            writer.write(record)
    writer.close()
    stats = writer.stats()
    part = {"rank": rank, "num_records": writer.num_records,
        "shards": [os.path.join(f"rank_{rank:05d}", x) for x in writer.shards],
        "shard_sizes": writer.shard_sizes, "field_bytes": stats["field_bytes"],
        "field_dtypes": stats["field_dtypes"]}
    os.makedirs(rank_dir, exist_ok=True)
    with open(os.path.join(rank_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(part, handle, indent=2)
    if world > 1: dist.barrier()
    if rank == 0:
        parts = []
        for worker in range(world):
            with open(os.path.join(args.cache_dir, f"rank_{worker:05d}", "manifest.json"), encoding="utf-8") as handle:
                parts.append(json.load(handle))
        raw_dim = len(layers) * adapter.hidden_size
        manifest = {"format": f"unified-parallel-draft-v1-{adapter.kind}",
            "target_type": adapter.kind, "num_records": sum(x["num_records"] for x in parts),
            "shards": sum((x["shards"] for x in parts), []),
            "shard_sizes": sum((x["shard_sizes"] for x in parts), []),
            "target_ckpt": os.path.realpath(args.target_ckpt), "dataset": args.dataset,
            "index_file": args.index_file, "target_layer_ids": layers,
            "item_token_num": args.item_token_num, "hidden_size": adapter.hidden_size,
            "target_context_dim": adapter.hidden_size if args.project_context else raw_dim,
            "raw_target_context_dim": raw_dim, "model_max_length": args.model_max_length,
            "cache_dtype": args.cache_dtype, "packed_shards": True, "teacher_cache": args.teacher_cache,
            "teacher_legal_sizes": [len(x) for x in legal_ids] if args.teacher_cache == "full_legal" else [],
            "field_bytes": merge(parts, "field_bytes"), "field_dtypes": merge(parts, "field_dtypes"),
            "context_is_projected": args.project_context,
            "projection_ckpt": os.path.realpath(args.draft_ckpt_for_projection) if args.project_context else None,
            "sample": sample_info}
        save_manifest(args.cache_dir, manifest)
        print(f"Saved {manifest['num_records']} {adapter.kind} records to {args.cache_dir}", flush=True)
    if world > 1: dist.destroy_process_group()


if __name__ == "__main__":
    main()
