#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <output_directory> <small_model_name> [fallback_threshold] [rollback_threshold]"
    exit 1
fi

output_directory=$1
small_model_name=$2
fallback_threshold=${3:-0.5}
rollback_threshold=${4:-1}

mkdir -p "$output_directory"

python src/run_translation.py \
    --output_dir "$output_directory" \
    --cache_dir /local1/hfs/gs_stuff/cache \
    --model_name_large kssteven/mT5-large-wmt2014-de-en \
    --model_name_small "$small_model_name" \
    --tokenizer_name google/mt5-small \
    --dataset_name lilferrit/wmt14-short \
    --source_lang de --target_lang en \
    --fallback_threshold $fallback_threshold \
    --rollback_threshold $rollback_threshold \
    --metric_for_best_model bleu \
    --num_beam 1 \
    --evaluation_strategy epoch --save_strategy epoch \
    --do_eval \
    --per_device_eval_batch_size 1 \
    --predict_with_generate