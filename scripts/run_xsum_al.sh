#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

fallback_threshold=${1:-0.4}
rollback_threshold=${2:-4}

echo "Running with fallback threshold: $fallback_threshold"
echo "Rollback threshold: $rollback_threshold"

python src/run_summarization.py \
    --output_dir /local1/hfs/gs_stuff/al-xsum-bild-res \
    --cache_dir /local1/hfs/gs_stuff/cache \
    --model_name_large kssteven/T5-large-xsum \
    --model_name_small paulh27/xsum_aligned_smallmT5 \
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
    --predict_with_generate \