#!/bin/bash

fallback_threshold=${1:-0.5}
rollback_threshold=${2:-1}
max_eval_samples=${3:-888}  # debug

python src/run_translation.py \
--output_dir ./out/aligned \
--model_name_large kssteven/mT5-large-iwslt2017-de-en --model_name_small kssteven/mT5-small-iwslt2017-de-en-bild-aligned \
--tokenizer_name google/mt5-small \
--dataset_name iwslt2017 --dataset_config_name iwslt2017-de-en \
--source_lang de --target_lang en \
--fallback_threshold $fallback_threshold \
--rollback_threshold $rollback_threshold \
--metrics sacrebleu meteor \
--num_beam 1 \
--evaluation_strategy epoch --save_strategy epoch \
--do_eval \
--per_device_eval_batch_size 1 \
--predict_with_generate \
--max_eval_samples $max_eval_samples
