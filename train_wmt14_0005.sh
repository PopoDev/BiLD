huggingface-cli login --token "$1"

export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false

python src/training/run_translation.py \
    --output_dir /local1/hfs/gs_stuff/ft-wmt14-5 \
    --cache_dir /local1/hfs/gs_stuff/cache \
    --model_name_or_path google/mt5-small \
    --resume_from_checkpoint /local1/hfs/gs_stuff/ft-wmt14-5/checkpoint-60000 \
    --tokenizer_name google/mt5-small \
    --dataset_name lilferrit/wmt14-short \
    --source_lang de \
    --target_lang en \
    --learning_rate 0.0005 \
    --gradient_accumulation_steps 2 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --predict_with_generate \
    --push_to_hub \
    --generation_max_length 150 \
    --dataloader_pin_memory False \
    --disable_tqdm False \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --evaluation_strategy steps \
    --save_strategy steps \
    --eval_steps 10000 \
    --save_total_limit 1 \
    --max_steps 100000 \
    --logging_steps 1000 \
    --save_steps 10000 \
    --load_best_model_at_end True \
    --metric_for_best_model eval_bleu \
    --greater_is_better True \
    --optim adafactor \
    --adafactor True \
    --lr_scheduler_type constant \