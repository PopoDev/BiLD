huggingface-cli login --token "$1"

export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false

learning_rates=("0.00005" "0.0001" "0.0002" "0.0005")

# Iterate over each learning rate
for lr in "${learning_rates[@]}"; do
    lr_dir=$(echo "${lr}" | sed 's/\./-/g')

    python src/run_translation.py \
        --output_dir "/local1/hfs/gs_stuff/ft-wmt14/${lr_dir}" \
        --cache_dir /local1/hfs/gs_stuff/cache \
        --model_name google-t5/t5-small \
        --tokenizer_name google-t5/t5-small \
        --dataset_name wmt14 \
        --dataset_config_name de-en \
        --source_lang de \
        --target_lang en \
        --learning_rate "${lr}" \
        --gradient_accumulation_steps 2 \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --save_total_limit 10 \
        --max_steps 5000 \
        --predict_with_generate \
        --push_to_hub \
        --generation_max_length 200 \
        --dataloader_pin_memory \
        --dataloader_num_workers 4 \
        --logging_steps 5000 \
        --disable_tqdm False \
        --save_strategy steps \
        --save_steps 5000 \
        --do_train \
        --source_prefix 'translate German to English: ' \
        --overwrite_output_dir \
        --evaluation_strategy steps \
        --eval_steps 5000 \
        --metric_for_best_model bleu \
        --load_best_model_at_end \

done
