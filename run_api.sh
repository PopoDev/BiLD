#!/bin/bash

model=${1:-vanilla}
fallback_threshold=${2:-0.5}
rollback_threshold=${3:-5}
max_eval_samples=${4:-2441}  # debug

case "$model" in
    unaligned)
        model_args="
        --model_name_large SEBIS/code_trans_t5_large_api_generation_transfer_learning_finetune \
        --model_name_small SEBIS/code_trans_t5_small_api_generation_transfer_learning_finetune \
        --fallback_threshold $fallback_threshold --rollback_threshold $rollback_threshold \
        --max_eval_samples $max_eval_samples"
        ;;
    vanilla)
        model_args="--model_name SEBIS/code_trans_t5_large_api_generation_transfer_learning_finetune"
        ;;
    *)
        echo "Invalid argument. Please use 'unaligned', or 'vanilla'."
        exit 1
        ;;
esac

python src/run_summarization.py $model_args \
    --output_dir ./out/$model \
    --tokenizer_name SEBIS/code_trans_t5_small_api_generation_transfer_learning_finetune \
    --dataset_name paulh27/java_code_api_generation \
    --source_prefix "description for api: " \
    --metrics rouge bleu \
    --num_beam 1 \
    --evaluation_strategy epoch --save_strategy epoch \
    --do_eval \
    --per_device_eval_batch_size 1 \
    --trust_remote_code True \
    --predict_with_generate