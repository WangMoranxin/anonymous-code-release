# APD-GR

APD-GR is an architecture-agnostic acceleration framework for generative recommendation. It distills the autoregressive knowledge of a frozen recommendation target into a lightweight draft model that predicts semantic-identifier positions in parallel.

Generative recommendation retrieves items by decoding semantic identifiers with beam search. Repeated target-model forward passes make this process expensive, especially for long identifiers and large beams. APD-GR reduces this cost without retraining or modifying the target model.

## Method

APD-GR uses two-stage autoregressive-to-parallel distillation:

1. **Parallel SID supervision** trains the draft to predict every semantic-identifier position concurrently.
2. **Prefix-distribution alignment** aligns the draft distributions with the frozen target under valid identifier prefixes.

The framework exposes one adapter API for both decoder-only and encoder-decoder targets. An adapter owns target execution, dataset semantics, legal identifier tokens, and item-code mappings; the parallel draft, cache format, loss, and constrained decoding remain target-independent.

In the accompanying study, APD-GR is evaluated on three datasets and five generative-recommendation backbones. It reaches a 9.22× average end-to-end inference speedup while largely preserving recommendation quality.

## Supported targets

| Target family | Adapter key |
| --- | --- |
| Causal language model / PEFT adapter | `causal` |
| MiniOneRec | `minionerec` |
| Standard T5 | `t5` |
| CoFiRec | `cofirec` |
| DIGER | `diger` |

`apd_gr.make_target_adapter(args)` is the public target-integration entry point. It returns an adapter with `load_tokenizer`, `load_model`, `extract_cache`, `prompt_features`, cache/data accessors, and identifier-constraint accessors.

## Installation

```bash
pip install -e ".[dev]"
```

Install only the target-specific extras you need when applicable:

```bash
pip install -e ".[diger]"
pip install -e ".[minionerec]"
pip install -e ".[causal-peft]"
```

## Workflow

1. Build a frozen-target feature cache:

```bash
apd-gr-cache --target_ckpt TARGET --target_type auto --data_path DATA --cache_dir CACHE
```

2. Train the parallel draft from that cache:

```bash
apd-gr-train --cache_dir CACHE --output_dir OUTPUT --target_ckpt TARGET --data_path DATA
```

3. Run constrained beam evaluation:

```bash
apd-gr-evaluate --target_ckpt TARGET --draft_ckpt OUTPUT --data_path DATA
```

Run `apd-gr-cache --help`, `apd-gr-train --help`, or `apd-gr-evaluate --help` for complete arguments. Checkpoints, datasets, caches, and experiment outputs are external inputs and are excluded from version control.

## Repository layout

```text
src/apd_gr/
  backends/       target-model execution implementations
  integrations/   recommendation data and identifier semantics
  scripts/        cache construction, training, and evaluation CLIs
  target_adapters.py  target registration and composition entry point
  modeling_parallel_draft.py  parallel draft architecture
```

## Reproducibility and release scope

The repository contains executable framework code only. Do not commit datasets, checkpoints, caches, virtual environments, or experiment logs. Before public release, add the final license and third-party attribution required for any upstream-derived integration implementation.
