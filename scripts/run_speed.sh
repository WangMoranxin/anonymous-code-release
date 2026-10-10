#!/usr/bin/env bash
set -euo pipefail

# The only experiment inputs to edit.
CUDA_VISIBLE_DEVICES=0
NPROC_PER_NODE=1
MASTER_PORT=3221
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
mkdir -p "$OUTPUT_ROOT/results"
TIMEFORMAT=$'wall_seconds=%3R\nuser_seconds=%3U\nsystem_seconds=%3S'
{ time CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" torchrun --nproc_per_node="$NPROC_PER_NODE" --master_port="$MASTER_PORT" -m apd_gr.scripts.evaluate \
    --seed 42 --target_type "$TARGET_TYPE" --target_ckpt "$TARGET_MODEL" --base_model "$BASE_MODEL" --tokenizer_path "$TOKENIZER_PATH" --data_path "$DATA_PATH" --dataset "$DATASET" --max_his_len 20 --model_max_length "$MODEL_MAX_LENGTH" --item_token_num 4 --target_layer_ids auto --target_config "$TARGET_CONFIG" \
    --draft_ckpt "$OUTPUT_ROOT/stage2_distill" --results_file "$OUTPUT_ROOT/results/speed.json" --debug_file "$OUTPUT_ROOT/results/speed_debug.json" --test_batch_size 1 --num_beams 50 --sample_num 1000 --test_prompt_ids 0 --bf16; } \
  2>&1 | tee "$OUTPUT_ROOT/results/speed.log"
