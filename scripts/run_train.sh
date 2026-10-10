#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=0,1
NPROC_PER_NODE=2
MASTER_PORT=3211
BACKBONE=letter-lc-rec
MODEL=Llama-3.2-1B
DATASET=Instruments
TARGET_MODEL=
DATA_PATH=

# letter-lc-rec requires BASE_MODEL; DIGER requires TARGET_CONFIG.
case "$BACKBONE" in
  letter-lc-rec) TARGET_TYPE=causal; BASE_MODEL=""; TOKENIZER_PATH=$TARGET_MODEL; MODEL_MAX_LENGTH=512; TARGET_CONFIG="" ;;
  letter-tiger) TARGET_TYPE=t5; BASE_MODEL=""; TOKENIZER_PATH=""; MODEL_MAX_LENGTH=512; TARGET_CONFIG="" ;;
  cofirec) TARGET_TYPE=cofirec; BASE_MODEL=""; TOKENIZER_PATH=""; MODEL_MAX_LENGTH=512; TARGET_CONFIG="" ;;
  diger) TARGET_TYPE=diger; BASE_MODEL=""; TOKENIZER_PATH=""; MODEL_MAX_LENGTH=210; TARGET_CONFIG="" ;;
  minionerec) TARGET_TYPE=minionerec; BASE_MODEL=""; TOKENIZER_PATH=$TARGET_MODEL; MODEL_MAX_LENGTH=2560; TARGET_CONFIG="" ;;
  *) echo "Unsupported BACKBONE: $BACKBONE" >&2; exit 2 ;;
esac

OUTPUT_ROOT=output/$BACKBONE/$MODEL/$DATASET
STAGE1_CACHE=$OUTPUT_ROOT/cache/stage1_root_only
STAGE1_DRAFT=$OUTPUT_ROOT/stage1_warmup
STAGE2_CACHE=$OUTPUT_ROOT/cache/stage2_full_legal
STAGE2_DRAFT=$OUTPUT_ROOT/stage2_distill

COMMON_ARGS=(--seed 42 --target_type "$TARGET_TYPE" --target_ckpt "$TARGET_MODEL" --base_model "$BASE_MODEL" --tokenizer_path "$TOKENIZER_PATH" --data_path "$DATA_PATH" --dataset "$DATASET" --max_his_len 20 --model_max_length "$MODEL_MAX_LENGTH" --item_token_num 4 --target_layer_ids auto --target_config "$TARGET_CONFIG")

if [[ ! -f "$STAGE1_CACHE/manifest.json" ]]; then
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" torchrun --nproc_per_node="$NPROC_PER_NODE" --master_port="$MASTER_PORT" -m apd_gr.scripts.prepare_cache \
    "${COMMON_ARGS[@]}" --cache_dir "$STAGE1_CACHE" --sample_fraction 0.5 --local_batch_size 64 --teacher_cache root_only --cache_dtype bfloat16 --bf16
fi
if [[ ! -f "$STAGE1_DRAFT/model.safetensors" ]]; then
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" torchrun --nproc_per_node="$NPROC_PER_NODE" --master_port="$((MASTER_PORT + 1))" -m apd_gr.scripts.train \
    "${COMMON_ARGS[@]}" --cache_dir "$STAGE1_CACHE" --output_dir "$STAGE1_DRAFT" --epochs 4 --per_device_batch_size 64 --gradient_accumulation_steps 1 --learning_rate 6e-4 --ce_alpha 1.0 --kl_alpha 0.0 --bf16
fi
if [[ ! -f "$STAGE2_CACHE/manifest.json" ]]; then
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" torchrun --nproc_per_node="$NPROC_PER_NODE" --master_port="$((MASTER_PORT + 2))" -m apd_gr.scripts.prepare_cache \
    "${COMMON_ARGS[@]}" --cache_dir "$STAGE2_CACHE" --local_batch_size 64 --teacher_cache full_legal --project_context --draft_ckpt_for_projection "$STAGE1_DRAFT" --cache_dtype bfloat16 --bf16
fi
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" torchrun --nproc_per_node="$NPROC_PER_NODE" --master_port="$((MASTER_PORT + 3))" -m apd_gr.scripts.train \
  "${COMMON_ARGS[@]}" --cache_dir "$STAGE2_CACHE" --output_dir "$STAGE2_DRAFT" --draft_init_ckpt "$STAGE1_DRAFT" --freeze_context_projection true --epochs 8 --per_device_batch_size 64 --gradient_accumulation_steps 1 --learning_rate 6e-4 --ce_alpha 0.8 --kl_alpha 0.2 --bf16
