#!/bin/bash

# Ensure the required arguments are provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <output_directory> <small_model_name> [fallback_threshold] [rollback_threshold]"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false

output_directory=$1
small_model_name=$2
fallback_threshold=${3:-0.4}
rollback_threshold=${4:-4}

mkdir -p "$output_directory"

python src/run_summarization.py \
    --output_dir "$output_directory" \
    --cache_dir /local1/hfs/gs_stuff/cache \
    --model_name_large kssteven/T5-large-xsum \
    --model_name_small "$small_model_name" \
    --tokenizer_name google-t5/t5-small \
    --dataset_name xsum \
    --source_prefix "summarize: " \
    --fallback_threshold $fallback_threshold \
    --rollback_threshold $rollback_threshold \
    --num_beam 1 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --do_eval \
    --per_device_eval_batch_size 1 \
    --trust_remote_code True \
    --predict_with_generate
