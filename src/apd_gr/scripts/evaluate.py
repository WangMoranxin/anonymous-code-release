import argparse
import json
import os
import re
import sys
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from apd_gr.modeling_parallel_draft import ParallelDraftModel
from apd_gr.utils import (add_common_args, build_seqrec_args, load_item_indices, load_repo_module, parse_int_list, semantic_token_id, set_seed)
from apd_gr.target_adapters import (apply_adapter_item_token_num, make_target_adapter,
                             resolve_target_layer_ids)



_ITEM_TOKEN_RE = re.compile(r"<[abcd]_[^>]+>")


def _extract_item_tokens(text: str):
    return _ITEM_TOKEN_RE.findall(str(text).replace(" ", ""))


def _prefix_hit_summary(target: str, predictions, item_token_num: int = 4):
    target_tokens = _extract_item_tokens(target)[:item_token_num]
    summary = {}
    for pos in range(item_token_num):
        prefix_len = pos + 1
        prefix = target_tokens[:prefix_len]
        summary[f"pos{pos}"] = bool(
            len(prefix) == prefix_len
            and any(_extract_item_tokens(pred)[:prefix_len] == prefix for pred in predictions)
        )
    return summary


def _hit_rank(hit_vector):
    for idx, value in enumerate(hit_vector):
        if int(value) != 0:
            return idx + 1
    return -1


def _should_save_debug(debug_records, limit: int) -> bool:
    return int(limit) < 0 or len(debug_records) < int(limit)


def build_result_payload(*, test_prompt_ids, metric_names, prompt_results,
                         num_beams, evaluation_seconds, evaluated_examples):
    """Build the single ordered result schema shared by every backbone."""
    if not prompt_results:
        raise ValueError("Cannot build evaluation results without prompt results")
    ordered_rows = [
        {metric: float(row[metric]) for metric in metric_names}
        for row in prompt_results
    ]
    mean_results, min_results, max_results = {}, {}, {}
    for metric in metric_names:
        values = [row[metric] for row in ordered_rows]
        mean_results[metric] = sum(values) / len(values)
        min_results[metric] = min(values)
        max_results[metric] = max(values)
    return {
        "test_prompt_ids": str(test_prompt_ids),
        "mean_results": mean_results,
        "num_beams": int(num_beams),
        "min_results": min_results,
        "max_results": max_results,
        "all_prompt_results": ordered_rows,
        "evaluation_seconds": float(evaluation_seconds),
        "samples_per_second": int(evaluated_examples) / max(float(evaluation_seconds), 1e-9),
    }


def save_result_payload(path, payload):
    """Write evaluation JSON without alphabetically reordering its schema."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = dict(self.dataset[index])
        item["dataset_index"] = int(index)
        return item

    def __getattr__(self, name):
        return getattr(self.dataset, name)


class IndexedTestCollator:
    def __init__(self, collator):
        self.collator = collator

    def __call__(self, batch):
        dataset_indices = [int(item["dataset_index"]) for item in batch]
        clean_batch = [
            {key: value for key, value in item.items() if key != "dataset_index"}
            for item in batch
        ]
        inputs, targets = self.collator(clean_batch)
        return inputs, targets, dataset_indices



def init_dist():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", local_rank))
    if world_size > 1 and not dist.is_initialized():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    return world_size, rank, local_rank


@torch.inference_mode()
def run_target_context(adapter, target, input_ids, attention_mask, target_layer_ids):
    target_context, root_hidden = adapter.prompt_features(
        target, input_ids, attention_mask, target_layer_ids
    )
    return root_hidden, target_context


def _last_valid_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    masked_positions = attention_mask.long() * positions.view(1, -1)
    return masked_positions.max(dim=1).values


def build_gpu_dfa(draft, indices: dict, tokenizer, item_token_num: int, device):
    """Compile the fixed-length item trie into dense per-position GPU transitions."""
    item_token_num = int(item_token_num)
    token_rows = []
    for code_tokens in indices.values():
        if len(code_tokens) != item_token_num:
            raise ValueError(
                f"Item code has {len(code_tokens)} positions; expected {item_token_num}"
            )
        token_rows.append(tuple(
            int(token) if isinstance(token, (int, float)) else semantic_token_id(tokenizer, token)
            for token in code_tokens
        ))

    prefix_maps = [{(): 0}]
    for depth in range(1, item_token_num + 1):
        prefixes = sorted({row[:depth] for row in token_rows})
        prefix_maps.append({prefix: idx for idx, prefix in enumerate(prefixes)})

    transitions = []
    for pos in range(item_token_num):
        global_ids = draft.draft_head.global_ids(pos).detach().cpu().tolist()
        global_to_local = {int(token_id): idx for idx, token_id in enumerate(global_ids)}
        table = torch.full(
            (len(prefix_maps[pos]), len(global_ids)),
            -1,
            dtype=torch.long,
        )
        for row in token_rows:
            parent_state = prefix_maps[pos][row[:pos]]
            local_id = global_to_local.get(int(row[pos]))
            if local_id is None:
                raise ValueError(
                    f"DFA token {row[pos]} at position {pos} is absent from the draft head"
                )
            next_state = prefix_maps[pos + 1][row[: pos + 1]]
            existing = int(table[parent_state, local_id].item())
            if existing not in (-1, next_state):
                raise ValueError("Non-deterministic item-code transition")
            table[parent_state, local_id] = next_state
        transitions.append(table.to(device=device, non_blocking=True))
    return transitions


@torch.inference_mode()
def batched_constrained_parallel_draft_beam(
    *,
    adapter,
    target,
    draft,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    dfa_transitions,
    target_layer_ids,
    num_beams: int,
    item_token_num: int,
    bias_model=None,
):
    device = input_ids.device
    bsz = input_ids.shape[0]
    beam_size = int(num_beams)
    item_token_num = int(item_token_num)

    root_hidden, target_context = run_target_context(
        adapter, target, input_ids, attention_mask, target_layer_ids
    )
    last_indices = _last_valid_indices(attention_mask)
    draft_hidden = draft.draft_hidden_for_item_tokens(
        target_hidden_states=target_context,
        prompt_last_position=last_indices,
        context_attention_mask=attention_mask,
        proposal_len=item_token_num,
    )

    if bias_model is not None:
        output_tokens, scores, _, _ = guided_beam_from_hidden(
            draft=draft, bias_model=bias_model, tokenizer=tokenizer,
            root_hidden=root_hidden, draft_hidden=draft_hidden,
            dfa_transitions=dfa_transitions, num_beams=num_beams,
            item_token_num=item_token_num,
        )
        return output_tokens, scores

    output_tokens = torch.full(
        (bsz, 1, item_token_num),
        int(tokenizer.pad_token_id),
        dtype=torch.long,
        device=device,
    )
    scores = torch.zeros((bsz, 1), dtype=draft_hidden.dtype, device=device)
    state = root_hidden.unsqueeze(1)
    dfa_states = torch.zeros((bsz, 1), dtype=torch.long, device=device)

    for pos in range(item_token_num):
        current_beam = output_tokens.shape[1]
        step_hidden = draft_hidden[:, pos, :].unsqueeze(1).expand(-1, current_beam, -1)
        local_logits = draft.draft_head.logits_for_position(
            state=state.reshape(bsz * current_beam, -1),
            hidden_states=step_hidden.reshape(bsz * current_beam, -1),
            pos=pos,
        ).float().view(bsz, current_beam, -1)
        local_vocab_size = local_logits.shape[-1]
        transition_rows = dfa_transitions[pos].index_select(
            0, dfa_states.reshape(-1)
        ).view(bsz, current_beam, local_vocab_size)
        local_logits = local_logits.masked_fill(transition_rows < 0, float("-inf"))
        local_log_probs = F.log_softmax(local_logits, dim=-1)
        expanded_scores = scores.unsqueeze(-1) + local_log_probs

        flat_scores = expanded_scores.flatten(1)
        next_beam = min(beam_size, flat_scores.shape[1])
        next_scores, flat_indices = torch.topk(
            flat_scores, k=next_beam, dim=1, largest=True, sorted=True
        )
        parent_indices = torch.div(
            flat_indices, local_vocab_size, rounding_mode="floor"
        )
        next_local_ids = flat_indices.remainder(local_vocab_size)
        global_ids = draft.draft_head.global_ids(pos).to(device)
        gather_tokens = parent_indices.unsqueeze(-1).expand(-1, -1, item_token_num)
        next_tokens = torch.gather(output_tokens, 1, gather_tokens).clone()
        next_tokens[:, :, pos] = global_ids.index_select(
            0, next_local_ids.reshape(-1)
        ).view(bsz, next_beam)
        dfa_states = torch.gather(
            transition_rows.flatten(1), 1, flat_indices
        )
        gather_state = parent_indices.unsqueeze(-1).expand(-1, -1, state.shape[-1])
        parent_state = torch.gather(state, 1, gather_state)
        output_tokens = next_tokens
        scores = next_scores.to(dtype=draft_hidden.dtype)

        if pos + 1 < item_token_num:
            state = draft.draft_head.update_state(
                state=parent_state.reshape(bsz * next_beam, -1),
                pos=pos,
                token_ids=output_tokens[:, :, pos].reshape(-1),
            ).view(bsz, next_beam, -1)
        else:
            state = parent_state

    if output_tokens.shape[1] < beam_size:
        pad_count = beam_size - output_tokens.shape[1]
        output_tokens = torch.cat(
            [output_tokens, output_tokens[:, -1:, :].expand(-1, pad_count, -1)], dim=1
        )
        scores = torch.cat([scores, scores[:, -1:].expand(-1, pad_count)], dim=1)
    return output_tokens[:, :beam_size], scores[:, :beam_size]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate parallel_draft with constrained beam search."
    )
    parser = add_common_args(parser)
    parser.add_argument("--draft_ckpt", type=str, required=True)
    parser.add_argument("--results_file", type=str, required=True)
    parser.add_argument("--debug_file", type=str, default="")
    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument("--num_beams", type=int, default=50)
    parser.add_argument("--sample_num", type=int, default=-1)
    parser.add_argument("--test_prompt_ids", type=str, default="0")
    parser.add_argument("--metrics", type=str, default="hit@1,hit@5,hit@10,hit@20,hit@50,ndcg@5,ndcg@10,ndcg@20,ndcg@50")
    parser.add_argument("--debug_sample_num", type=int, default=20)
    parser.add_argument("--reranker_checkpoint", type=str, default="")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--fp16", action="store_true", default=False)
    return parser.parse_args()


class IntegerCodeTestCollator:
    def __init__(self, adapter):
        self.cache_collator = adapter.cache_collator(None)

    def __call__(self, rows):
        packed = self.cache_collator(rows)
        return {"input_ids": packed["input_ids"], "attention_mask": packed["attention_mask"]}, packed["target_ids"]


@torch.inference_mode()
def evaluate_integer_codes(args, adapter, target, draft, tokenizer, target_layer_ids,
                   device, world_size, rank, bias_model=None):
    dataset = adapter.test_dataset()
    if int(args.sample_num) > 0:
        dataset = Subset(dataset, range(min(int(args.sample_num), len(dataset))))
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False) if world_size > 1 else None
    loader = DataLoader(dataset, batch_size=args.test_batch_size, sampler=sampler,
        shuffle=False, collate_fn=IntegerCodeTestCollator(adapter), num_workers=2, pin_memory=True)
    dfa = build_gpu_dfa(draft, adapter.item_code_dict(), tokenizer,
                        args.item_token_num, device)
    metric_names = [name.strip() for name in args.metrics.split(",") if name.strip()]
    for name in metric_names:
        family, separator, cutoff = name.partition("@")
        if family.lower() not in {"hit", "recall", "ndcg"} or not separator or not cutoff.isdigit():
            raise ValueError(f"Unsupported integer-code metric: {name}")
        if int(cutoff) > int(args.num_beams):
            raise ValueError(f"Metric {name} exceeds num_beams={args.num_beams}")
    totals = {name: 0.0 for name in metric_names}
    count = 0
    started = time.perf_counter()
    for inputs, labels in tqdm(loader, disable=rank != 0, desc=f"{adapter.kind}-gpu-dfa"):
        input_ids = inputs["input_ids"].to(device, non_blocking=True)
        attention_mask = inputs["attention_mask"].to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        predictions, _ = batched_constrained_parallel_draft_beam(
            adapter=adapter, target=target, draft=draft, tokenizer=tokenizer,
            input_ids=input_ids, attention_mask=attention_mask,
            dfa_transitions=dfa, target_layer_ids=target_layer_ids,
            num_beams=args.num_beams, item_token_num=args.item_token_num,
            bias_model=bias_model)
        matches = predictions.eq(labels.unsqueeze(1)).all(dim=-1)
        for name in metric_names:
            family, cutoff = name.lower().split("@", 1)
            k = int(cutoff)
            prefix = matches[:, :k]
            if family in {"hit", "recall"}:
                totals[name] += prefix.any(dim=1).float().sum().item()
            else:
                ranks = torch.arange(1, prefix.shape[1] + 1, device=device).view(1, -1)
                discounts = torch.where(prefix, 1.0 / torch.log2(ranks.float() + 1.0), 0.0)
                totals[name] += discounts.max(dim=1).values.sum().item()
        count += labels.shape[0]
    values = torch.tensor([count] + [totals[key] for key in metric_names],
                          dtype=torch.float64, device=device)
    if world_size > 1:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    if rank == 0:
        total_count = max(int(values[0].item()), 1)
        metrics = {key: float(values[idx + 1].item() / total_count)
                   for idx, key in enumerate(metric_names)}
        evaluation_seconds = time.perf_counter() - started
        payload = build_result_payload(
            test_prompt_ids=args.test_prompt_ids,
            metric_names=metric_names,
            prompt_results=[metrics],
            num_beams=args.num_beams,
            evaluation_seconds=evaluation_seconds,
            evaluated_examples=total_count,
        )
        save_result_payload(args.results_file, payload)
        print(json.dumps(payload, indent=2), flush=True)
    if world_size > 1:
        dist.barrier()


def main():
    args = parse_args()
    set_seed(args.seed)
    world_size, rank, local_rank = init_dist()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    adapter = make_target_adapter(args)
    apply_adapter_item_token_num(args, adapter)
    target_layer_ids = resolve_target_layer_ids(args.target_layer_ids, adapter.num_layers)
    args.target_layer_ids = ",".join(map(str, target_layer_ids))
    if rank == 0:
        print(f"Resolved target layers: {target_layer_ids}", flush=True)
    tokenizer = adapter.load_tokenizer(padding_side="right")
    target = adapter.load_model(
        tokenizer, device, torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    )
    draft = ParallelDraftModel.from_pretrained(
        args.draft_ckpt,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
    ).to(device).eval()
    checkpoint_layer_ids = [int(x) for x in draft.config.target_layer_ids]
    if checkpoint_layer_ids != target_layer_ids:
        raise ValueError(
            "Layer mismatch: draft checkpoint target_layer_ids="
            f"{checkpoint_layer_ids}, but --target_layer_ids={target_layer_ids}."
        )

    bias_model = None
    if args.reranker_checkpoint:
        payload = torch.load(args.reranker_checkpoint, map_location="cpu", weights_only=True)
        config = payload.get("config", {})
        bias_model = IncrementalSIDBias(
            hidden_size=draft.config.hidden_size,
            rank_dim=int(config.get("rank_dim", 64)),
            item_token_num=int(config.get("item_token_num", args.item_token_num)),
            alpha_max=float(config.get("alpha_max", .10)),
        ).to(device)
        bias_model.load_state_dict(payload["model"])
        bias_model.eval()
        if rank == 0:
            print(f"Loaded reranker: {args.reranker_checkpoint}", flush=True)

    if adapter.uses_integer_codes:
        evaluate_integer_codes(args, adapter, target, draft, tokenizer, target_layer_ids,
                       device, world_size, rank, bias_model=bias_model)
        if world_size > 1:
            dist.destroy_process_group()
        return

    data_module = load_repo_module(args.repo_root, "data.py")
    collator_module = load_repo_module(args.repo_root, "collator.py")
    metric_module = load_repo_module(args.repo_root, "evaluate.py")
    SeqRecDataset = data_module.SeqRecDataset
    TestCollator = collator_module.TestCollator
    get_topk_results = metric_module.get_topk_results
    get_metrics_results = metric_module.get_metrics_results
    seq_args = build_seqrec_args(args)
    seq_args.test_task = "SeqRec"
    base_test_data = SeqRecDataset(seq_args, mode="test", sample_num=args.sample_num)
    test_data = IndexedDataset(base_test_data)
    item_indices = load_item_indices(args.data_path, args.dataset, args.index_file)
    dfa_transitions = build_gpu_dfa(
        draft, item_indices, tokenizer, args.item_token_num, device
    )
    if rank == 0:
        state_counts = [int(table.shape[0]) for table in dfa_transitions]
        print(f"GPU DFA ready: state_counts={state_counts}", flush=True)
    all_items = test_data.get_all_items()
    sampler = DistributedSampler(test_data, shuffle=False, drop_last=False) if world_size > 1 else None
    loader = DataLoader(
        test_data,
        batch_size=args.test_batch_size,
        sampler=sampler,
        shuffle=False,
        collate_fn=IndexedTestCollator(TestCollator(seq_args, tokenizer)),
        num_workers=2,
        pin_memory=True,
    )

    prompt_ids = [0] if args.test_prompt_ids == "all" else [
        int(part) for part in args.test_prompt_ids.split(",") if part
    ]
    if adapter.kind == "cofirec" and prompt_ids != [0]:
        raise ValueError("CoFiRec SeqRec currently supports only test prompt id 0")

    metrics = args.metrics.split(",")
    all_prompt_results = []
    debug_records = []
    evaluated_examples = 0
    evaluation_start = time.perf_counter()
    for prompt_id in prompt_ids:
        test_data.set_prompt(prompt_id)
        metrics_results = {}
        total = 0
        iterator = tqdm(loader, disable=rank != 0, desc=f"prompt-{prompt_id}")
        for step, batch in enumerate(iterator):
            inputs = batch[0].to(device)
            targets = batch[1]
            dataset_indices = batch[2]
            output_ids, scores = batched_constrained_parallel_draft_beam(
                adapter=adapter,
                target=target,
                draft=draft,
                tokenizer=tokenizer,
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                dfa_transitions=dfa_transitions,
                target_layer_ids=target_layer_ids,
                num_beams=args.num_beams,
                item_token_num=args.item_token_num,
                bias_model=bias_model,
            )
            flat_output_ids = output_ids.reshape(-1, output_ids.shape[-1])
            batch_outputs = tokenizer.batch_decode(flat_output_ids, skip_special_tokens=True)
            batch_scores = scores.reshape(-1).detach().float().cpu().tolist()
            topk_res = get_topk_results(
                batch_outputs,
                batch_scores,
                targets,
                args.num_beams,
                all_items=all_items,
            )
            bs = len(targets)
            if world_size > 1:
                gathered_bs = [None for _ in range(world_size)]
                gathered_res = [None for _ in range(world_size)]
                dist.all_gather_object(gathered_bs, bs)
                dist.all_gather_object(gathered_res, topk_res)
                total += sum(gathered_bs)
                if rank == 0:
                    merged = []
                    for part in gathered_res:
                        merged.extend(part)
                    batch_metrics = get_metrics_results(merged, metrics)
                else:
                    batch_metrics = {}
            else:
                total += bs
                batch_metrics = get_metrics_results(topk_res, metrics)

            if rank == 0:
                for key, value in batch_metrics.items():
                    metrics_results[key] = metrics_results.get(key, 0.0) + value
            if _should_save_debug(debug_records, args.debug_sample_num):
                for sample_idx, target_item in enumerate(targets):
                    if not _should_save_debug(debug_records, args.debug_sample_num):
                        break
                    start = sample_idx * args.num_beams
                    end = start + args.num_beams
                    predictions = batch_outputs[start:end]
                    hit_vector = [int(x) for x in topk_res[sample_idx]]
                    debug_records.append(
                        {
                            "prompt_id": int(prompt_id),
                            "dataset_index": int(dataset_indices[sample_idx]),
                            "target": target_item,
                            "predictions": predictions,
                            "scores": batch_scores[start:end],
                            "hit_vector": hit_vector,
                            "hit_rank": _hit_rank(hit_vector),
                            "top1": predictions[0] if predictions else None,
                            "top1_hit": bool(hit_vector[0]) if hit_vector else False,
                            "target_in_beams": target_item in predictions,
                            "prefix_hit": _prefix_hit_summary(
                                target_item, predictions, args.item_token_num
                            ),
                        }
                    )
        if rank == 0:
            evaluated_examples += total
            prompt_result = {
                key: value / max(total, 1) for key, value in metrics_results.items()
            }
            all_prompt_results.append(prompt_result)
            print(f"Prompt {prompt_id} results: {prompt_result}")
        if world_size > 1:
            dist.barrier()

    if rank == 0:
        evaluation_seconds = time.perf_counter() - evaluation_start
        payload = build_result_payload(
            test_prompt_ids=args.test_prompt_ids,
            metric_names=metrics,
            prompt_results=all_prompt_results,
            num_beams=args.num_beams,
            evaluation_seconds=evaluation_seconds,
            evaluated_examples=evaluated_examples,
        )
        save_result_payload(args.results_file, payload)
        if world_size > 1:
            debug_gather_list = [None for _ in range(world_size)]
            dist.all_gather_object(debug_gather_list, debug_records)
            merged_debug_records = []
            for part in debug_gather_list:
                merged_debug_records.extend(part or [])
        else:
            merged_debug_records = list(debug_records)
        merged_debug_records.sort(
            key=lambda item: (int(item["prompt_id"]), int(item["dataset_index"]))
        )
        if int(args.debug_sample_num) >= 0:
            merged_debug_records = merged_debug_records[: int(args.debug_sample_num)]

        debug_file = args.debug_file or os.path.splitext(args.results_file)[0] + "_debug.json"
        with open(debug_file, "w", encoding="utf-8") as f:
            json.dump(merged_debug_records, f, indent=2)
        print(f"Saved results to {args.results_file}")
        print(f"Saved debug records to {debug_file}")
        print(
            f"Evaluation runtime: {evaluation_seconds:.3f}s, "
            f"throughput: {payload['samples_per_second']:.3f} samples/s"
        )
    elif world_size > 1:
        dist.all_gather_object([None for _ in range(world_size)], debug_records)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
