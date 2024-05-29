#!/bin/bash

fallback_threshold=${1:-0.6}
rollback_threshold=${2:-2}
max_predict_samples=${3:-8079}  # debug

python src/run_translation.py \
    --output_dir ./out/quality \
    --model_name_large kssteven/mT5-large-iwslt2017-de-en --model_name_small kssteven/mT5-small-iwslt2017-de-en \
    --tokenizer_name google/mt5-small \
    --dataset_name iwslt2017 --dataset_config_name iwslt2017-de-en \
    --source_lang de --target_lang en \
    --fallback_threshold $fallback_threshold \
    --rollback_threshold $rollback_threshold \
    --metric_for_best_model bleu \
    --num_beam 1 \
    --do_predict \
    --predict_with_generate \
    --per_device_eval_batch_size 1 \
    --trust_remote_code \
    --max_predict_samples $max_predict_samples
