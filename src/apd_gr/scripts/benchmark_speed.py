#!/usr/bin/env python3
"""Paired single-GPU target and APD-GR inference benchmark."""
import argparse
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from apd_gr.modeling_parallel_draft import ParallelDraftModel
from apd_gr.scripts.evaluate import (
    IntegerCodeTestCollator,
    batched_constrained_parallel_draft_beam,
    build_gpu_dfa,
)
from apd_gr.target_adapters import apply_adapter_item_token_num, make_target_adapter
from apd_gr.utils import (
    add_common_args,
    build_seqrec_args,
    load_item_indices,
    load_repo_module,
    semantic_token_id,
    set_seed,
)


def timing_payload(times):
    total = sum(times)
    return dict(samples=len(times), batches=len(times), batch_size=1,
                total_inference_seconds=total, mean_sample_seconds=total / len(times),
                mean_sample_ms=1000 * total / len(times), samples_per_second=len(times) / total,
                first_batch_seconds=times[0], batch_seconds=times)


def build_positional_constraints(codes, item_token_num):
    allowed_tokens = [set() for _ in range(item_token_num)]
    for code in codes:
        if len(code) != item_token_num:
            raise ValueError(
                f"SID length mismatch: expected {item_token_num}, got {len(code)}"
            )
        for pos, token in enumerate(code):
            allowed_tokens[pos].add(token)
    return tuple(frozenset(tokens) for tokens in allowed_tokens)


def positional_constraint(allowed_tokens, prompt_length):
    def allowed(_batch_id, sequence):
        position = sequence.shape[-1] - prompt_length
        if position < 0 or position >= len(allowed_tokens):
            raise ValueError(f"No legal SID tokens at generated position {position}")
        return list(allowed_tokens[position])
    return allowed


@torch.inference_mode()
def main():
    parser = add_common_args(argparse.ArgumentParser(description="Paired single-GPU target and APD-GR inference benchmark."))
    parser.add_argument('--draft_ckpt', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--sample_num', type=int, default=1000)
    parser.add_argument('--num_beams', type=int, default=50)
    parser.add_argument('--warmup_batches', type=int, default=5)
    parser.add_argument('--dtype', choices=['bf16', 'fp16', 'fp32'], default='bf16')
    parser.add_argument('--benchmark_name', default='apd-gr')
    args = parser.parse_args()
    if int(os.environ.get('WORLD_SIZE', '1')) != 1 or torch.cuda.device_count() != 1:
        raise ValueError('Expose exactly one GPU; distributed execution is disabled')
    if args.sample_num < 1 or args.num_beams < 1 or args.warmup_batches < 0:
        raise ValueError('Invalid sample/beam/warmup count')
    set_seed(args.seed)
    device = torch.device('cuda:0')
    dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}[args.dtype]
    adapter = make_target_adapter(args)
    apply_adapter_item_token_num(args, adapter)
    tokenizer = adapter.load_tokenizer(padding_side='right' if adapter.is_encoder_decoder else 'left')
    target = adapter.load_model(tokenizer, device, dtype)
    draft = ParallelDraftModel.from_pretrained(args.draft_ckpt, torch_dtype=dtype).to(device).eval()
    layers = list(map(int, draft.config.target_layer_ids))
    if int(draft.config.item_token_num) != args.item_token_num:
        raise ValueError('Draft/target SID length mismatch')
    if adapter.uses_integer_codes:
        dataset = adapter.test_dataset()
        collator = IntegerCodeTestCollator(adapter)
        indices = adapter.item_code_dict()
    else:
        seq_args = build_seqrec_args(args)
        dataset = load_repo_module(args.repo_root, 'data.py').SeqRecDataset(seq_args, mode='test', sample_num=-1)
        dataset.set_prompt(0)
        collator = load_repo_module(args.repo_root, 'collator.py').TestCollator(seq_args, tokenizer)
        indices = load_item_indices(args.data_path, args.dataset, args.index_file)
    if len(dataset) < args.sample_num:
        raise ValueError(f'Requested {args.sample_num} samples, dataset has {len(dataset)}')
    # Independent RNG: model loading cannot change the chosen sample IDs.
    import numpy as np
    sample_ids = np.random.default_rng(args.seed).choice(len(dataset), args.sample_num, replace=False).tolist()
    loader = DataLoader(Subset(dataset, sample_ids), batch_size=1, shuffle=False,
                        collate_fn=collator, num_workers=0)
    codes = [tuple(int(t) if isinstance(t, (int, float)) else semantic_token_id(tokenizer, t)
                   for t in row) for row in indices.values()]
    positional_tokens = build_positional_constraints(codes, args.item_token_num)
    dfa = build_gpu_dfa(draft, indices, tokenizer, args.item_token_num, device)
    metrics_k = sorted({k for k in (1, 5, 10, 20, 50) if k <= args.num_beams} | {args.num_beams})
    times = {'baseline': [], 'apd_gr': []}
    totals = {name: {f'{family}@{k}': 0.0 for family in ('hit', 'ndcg') for k in metrics_k}
              for name in times}
    fingerprint = hashlib.sha256()
    baseline_scope = 'constrained target.generate (including encoder/prompt forward)'

    def baseline(inputs, *, allowed, offset):
        if adapter.kind == 'diger':
            target.num_beams = args.num_beams
            return target.generate(**inputs, n_return_sequences=args.num_beams,
                                   prefix_allowed_tokens_fn=allowed)
        result = target.generate(**inputs, max_new_tokens=args.item_token_num,
                                 num_beams=args.num_beams, num_return_sequences=args.num_beams,
                                 do_sample=False, prefix_allowed_tokens_fn=allowed,
                                 eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                                 return_dict_in_generate=True, output_scores=True, early_stopping=True)
        return result.sequences[:, offset:offset + args.item_token_num].reshape(1, args.num_beams, -1)

    def fast(inputs):
        return batched_constrained_parallel_draft_beam(
            adapter=adapter, target=target, draft=draft, tokenizer=tokenizer,
            **inputs, dfa_transitions=dfa, target_layer_ids=layers,
            num_beams=args.num_beams, item_token_num=args.item_token_num)[0]

    for step, batch in enumerate(tqdm(loader, desc=f'{args.benchmark_name}/{args.dataset}')):
        inputs = {key: batch[0][key].to(device) for key in ('input_ids', 'attention_mask')}
        # Bind the sample's prompt offset before warmup and timed inference.
        offset = 1 if adapter.is_encoder_decoder else inputs['input_ids'].shape[1]
        allowed = positional_constraint(positional_tokens, offset)
        run_baseline = partial(baseline, allowed=allowed, offset=offset)
        labels = batch[1]
        if adapter.uses_integer_codes:
            if isinstance(labels, torch.Tensor):
                label = labels[0].tolist()
            else:
                label = labels[0]
                if isinstance(label, str):
                    label = [semantic_token_id(tokenizer, token) for token in __import__('re').findall(r'<[^>]+>', label)]
        else:
            label = [semantic_token_id(tokenizer, token) for token in __import__('re').findall(r'<[^>]+>', labels[0])]
        fingerprint.update(json.dumps([inputs['input_ids'].tolist(), inputs['attention_mask'].tolist(), label]).encode())
        if step == 0:
            for function in (run_baseline, fast):
                for _ in range(args.warmup_batches):
                    function(inputs)
                torch.cuda.synchronize(device)
        # Alternate order to reduce systematic clock/temperature order bias.
        order = [('baseline', run_baseline), ('apd_gr', fast)]
        if step % 2:
            order.reverse()
        for name, function in order:
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            predictions = function(inputs)
            torch.cuda.synchronize(device)
            times[name].append(time.perf_counter() - started)
            rows = predictions[0].tolist()
            for k in metrics_k:
                rank = next((i + 1 for i, row in enumerate(rows[:k]) if list(row) == list(label)), None)
                totals[name][f'hit@{k}'] += float(rank is not None)
                if rank:
                    import math
                    totals[name][f'ndcg@{k}'] += 1 / math.log2(rank + 1)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    settings = dict(vars(args), batch_size=1, gpu=torch.cuda.get_device_name(device),
                    cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    sample_ids=sample_ids, input_sha256=fingerprint.hexdigest(),
                    target_layer_ids=layers, torch_version=torch.__version__,
                    transformers_version=__import__('transformers').__version__,
                    cuda_version=torch.version.cuda,
                    deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                    deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
                    matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                    timer_scope='synchronized inference function wall time; warmup excluded',
                    excluded=['model/data/DFA loading', 'position-wise constraint preparation and prompt-offset binding',
                              'input transfer', 'decode/metrics', 'serialization'],
                    baseline_max_new_tokens=args.item_token_num,
                    generation_protocol='SID tokens only; no extra EOS decoding step',
                    constraints=('target uses position-wise legal SID tokens; APD-GR uses the legal-SID DFA; '
                                 'prompt 0; no OOM beam reduction'),
                    baseline_scope=baseline_scope,
                    baseline_note='DIGER uses its constrained GenerationMixin path, rather than its unconstrained native my_beam_search')
    payloads = {}
    for name in times:
        payload = dict(method=name, **timing_payload(times[name]), settings=settings)
        payloads[name] = payload
        (output / f'{name}_timing.json').write_text(json.dumps(payload, indent=2))
        (output / f'{name}_metrics.json').write_text(json.dumps({k: v / args.sample_num for k, v in totals[name].items()}, indent=2))
    summary = dict(backbone=args.benchmark_name, dataset=args.dataset, num_beams=args.num_beams, samples=args.sample_num,
                   baseline_ms=payloads['baseline']['mean_sample_ms'],
                   apd_gr_ms=payloads['apd_gr']['mean_sample_ms'],
                   speedup= sum(times['baseline']) / sum(times['apd_gr']), settings=settings)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != 'settings'}, indent=2))


if __name__ == '__main__':
    main()
