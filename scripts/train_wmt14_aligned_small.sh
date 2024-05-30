huggingface-cli login --token "$1"

export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false

python src/run_translation.py \
    --output_dir /local1/hfs/gs_stuff/al-wmt14 \
    --cache_dir /local1/hfs/gs_stuff/cache \
    --model_name google-t5/t5-small \
    --tokenizer_name google-t5/t5-small \
    --dataset_name paulh27/alignment_wmt2014_de_en \
    --source_lang de \
    --target_lang en \
    --learning_rate 0.0005 \
    --gradient_accumulation_steps 2 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --save_total_limit 10 \
    --max_steps 100000 \
    --predict_with_generate \
    --push_to_hub \
    --generation_max_length 200 \
    --dataloader_pin_memory \
    --dataloader_num_workers 4 \
    --logging_steps 5000 \
    --disable_tqdm False \
    --save_strategy steps \
    --save_steps 10000 \
    --do_train \
    --overwrite_output_dir \
    --evaluation_strategy steps \
    --eval_steps 10000 \
    --metric_for_best_model bleu \
    --load_best_model_at_end \
    --overwrite_output_dir