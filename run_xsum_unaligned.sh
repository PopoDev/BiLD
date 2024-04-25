#!/bin/bash

fallback_threshold=${1:-0.4}
rollback_threshold=${2:-4}
max_eval_samples=${3:-11300}  # debug

python src/run_summarization.py \
--output_dir ./out/unaligned \
--model_name_large kssteven/T5-large-xsum --model_name_small kssteven/T5-small-xsum \
--tokenizer_name google-t5/t5-small \
--dataset_name xsum \
--source_prefix "summarize: " \
--fallback_threshold $fallback_threshold \
--rollback_threshold $rollback_threshold \
--num_beam 1 \
--evaluation_strategy epoch --save_strategy epoch \
--do_eval \
--per_device_eval_batch_size 1 \
--trust_remote_code True \
--predict_with_generate \
--max_eval_samples $max_eval_samples