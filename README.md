# APD-GR

APD-GR is an architecture-agnostic acceleration framework for generative recommendation. It distills the autoregressive knowledge of a frozen recommendation target into a lightweight draft model that predicts semantic-identifier positions in parallel.

Generative recommendation retrieves items by decoding semantic identifiers with beam search. Repeated target-model forward passes make this process expensive, especially for long identifiers and large beams. APD-GR reduces this cost without retraining or modifying the target model.

## Method

APD-GR uses two-stage autoregressive-to-parallel distillation:

1. **Parallel SID supervision** trains the draft to predict every semantic-identifier position concurrently.
2. **Prefix-distribution alignment** aligns the draft distributions with the frozen target under valid identifier prefixes.

The framework exposes one adapter API for both decoder-only and encoder-decoder targets. An adapter owns target execution, dataset semantics, legal identifier tokens, and item-code mappings; the parallel draft, cache format, loss, and constrained decoding remain target-independent.

In the accompanying study, APD-GR is evaluated on three datasets and five generative-recommendation backbones. It reaches a 9.22× average end-to-end inference speedup while largely preserving recommendation quality.

## Installation

Create and activate a Python environment, then install the recorded dependencies from the repository root:

```bash
pip install -r requirements.txt
```

`requirements.txt` contains `-e .`, which installs APD-GR itself from the current repository. Use a CUDA/PyTorch build compatible with your GPUs. If your platform needs a different PyTorch build, install that build first and then install the remaining dependencies.

## Configure an experiment

At the beginning of each script in `scripts/`, set the following values for your environment:

```bash
CUDA_VISIBLE_DEVICES=0,1
NPROC_PER_NODE=2
MASTER_PORT=3211
BACKBONE=letter-lc-rec
MODEL=Llama-3.2-1B
DATASET=Instruments
TARGET_MODEL=/path/to/target-model
DATA_PATH=/path/to/dataset
```

For `letter-lc-rec`, also set `BASE_MODEL` in the corresponding backbone branch. For `diger`, set `TARGET_CONFIG`. The scripts save all generated artifacts below:

```text
output/<BACKBONE>/<MODEL>/<DATASET>/
```

## Two-stage training

After setting the inputs in `scripts/run_train.sh`, run:

```bash
bash scripts/run_train.sh
```

The script runs the full two-stage procedure:

1. Construct a root-only target cache and train the stage-1 warm-up draft.
2. Construct a legal-prefix target cache and train the stage-2 distillation draft initialized from stage 1.

The final draft checkpoint is written to:

```text
output/<BACKBONE>/<MODEL>/<DATASET>/stage2_distill/
```

## Evaluation

Set the same experiment inputs in `scripts/run_evaluate.sh`, then run:

```bash
bash scripts/run_evaluate.sh
```

The script evaluates the stage-2 draft with constrained beam search and writes metrics to:

```text
output/<BACKBONE>/<MODEL>/<DATASET>/results/evaluate.json
```

## Speed measurement

Set the same experiment inputs in `scripts/run_speed.sh`, then run:

```bash
bash scripts/run_speed.sh
```

The script evaluates 1,000 examples and records wall-clock timing in:

```text
output/<BACKBONE>/<MODEL>/<DATASET>/results/speed.log
```
