huggingface-cli login --token "$1"

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

python src/training/run_summarization.py \
    --output_dir /local1/hfs/gs_stuff/al-xsum \
    --cache_dir /local1/hfs/gs_stuff/cache \
    --resume_from_checkpoint /local1/hfs/gs_stuff/al-xsum/checkpoint-20000 \
    --model_name_or_path google-t5/t5-small \
    --tokenizer_name google-t5/t5-small \
    --dataset_name lilferrit/xsum_t5_distillation \
    --text_column document \
    --summary_column t5_large_output \
    --learning_rate 0.0002 \
    --gradient_accumulation_steps 2 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --save_total_limit 10 \
    --max_steps 100000 \
    --predict_with_generate \
    --push_to_hub \
    --generation_max_length 200 \
    --dataloader_pin_memory \
    --dataloader_num_workers 4 \
    --logging_steps 1000 \
    --disable_tqdm False \
    --save_strategy steps \
    --save_steps 10000 \
    --do_train \
    --source_prefix 'summarize: ' \
    --overwrite_output_dir \
    --evaluation_strategy steps \
    --eval_steps 10000 \
    --metric_for_best_model rougeL \
    --adafactor \
    --load_best_model_at_end \
