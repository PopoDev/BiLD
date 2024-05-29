#!/bin/bash

# Log in to Hugging Face CLI with the provided token
huggingface-cli login --token "$1"

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# Hardcoded rollback threshold value
rollback_threshold=${2:-1}

# Array of fallback threshold values
fallback_thresholds=(0.1 0.2 0.3 0.4 0.5)

# Iterate over each fallback threshold and run the translation script
for fallback_threshold in "${fallback_thresholds[@]}"; do
    fallback_threshold_dir=$(echo "$fallback_threshold" | sed 's/\./-/')
    rollback_threshold_dir=$(echo "$rollback_threshold" | sed 's/\./-/')

    output_dir="/local1/hfs/gs_stuff/al-wmt14-bild-res/${fallback_threshold_dir}_${rollback_threshold_dir}"

    # Create the output directory if it doesn't exist
    mkdir -p "$output_dir"
    
    echo "Running with fallback threshold: $fallback_threshold"
    echo "Rollback threshold: $rollback_threshold"
    echo "Output directory: $output_dir"
    
    # Run the translation script with the current fallback threshold
    python src/run_translation.py \
        --output_dir "$output_dir" \
        --model_name_large kssteven/mT5-large-wmt2014-de-en \
        --model_name_small lilferrit/al-wmt14 \
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
done
