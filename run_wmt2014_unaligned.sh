#!/bin/bash

fallback_threshold=${1:-0.5}
rollback_threshold=${2:-1}

huggingface-cli login --token "$1"

export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false

python src/run_translation.py \
    --output_dir /local1/hfs/gs_stuff/al-wmt14-bild-res \
    --model_name_large kssteven/mT5-large-wmt2014-de-en \
    --model_name_small kssteven/mT5-small-wmt2014-de-en \
    --tokenizer_name google/mt5-small \
    --dataset_name wmt14 --dataset_config_name de-en \
    --source_lang de --target_lang en \
    --fallback_threshold $fallback_threshold \
    --rollback_threshold $rollback_threshold \
    --metric_for_best_model bleu \
    --num_beam 1 \
    --evaluation_strategy epoch --save_strategy epoch \
    --do_eval \
    --per_device_eval_batch_size 1 \
    --predict_with_generate \