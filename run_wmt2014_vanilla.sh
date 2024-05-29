#!/bin/bash

DEFAULT_OUTPUT_DIR="/local1/hfs/gs_stuff/vn-wmt14-bild-res"
DEFAULT_MODEL_NAME="kssteven/mT5-large-wmt2014-de-en"

OUTPUT_DIR="${1:-$DEFAULT_OUTPUT_DIR}"
MODEL_NAME="${2:-$DEFAULT_MODEL_NAME}"

export CUDA_VISIBLE_DEVICES=1

echo "Output Directory: $OUTPUT_DIR"
echo "Model Name: $MODEL_NAME"

python src/run_translation.py \
    --output_dir "$OUTPUT_DIR" \
    --model_name "$MODEL_NAME" \
    --cache_dir /local1/hfs/gs_stuff/cache \
    --tokenizer_name google/mt5-small \
    --dataset_name lilferrit/wmt14-short \
    --source_lang de --target_lang en \
    --metric_for_best_model bleu \
    --num_beam 1 \
    --evaluation_strategy epoch --save_strategy epoch \
    --do_eval \
    --per_device_eval_batch_size 1 \
    --predict_with_generate
