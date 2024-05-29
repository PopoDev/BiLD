huggingface-cli login --token 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
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false

python src/training/run_summarization.py \
--output_dir /tmp/finetuned_models/xsum_aligned_smallT5_cont3 \
--cache_dir /tmp/cache \
--model_name_or_path /tmp/finetuned_models/xsum_aligned_smallT5_cont2/checkpoint-60000 \
--tokenizer_name google-t5/t5-small \
--dataset_name lilferrit/xsum_t5_distillation \
--text_column document \
--summary_column t5_large_output \
--learning_rate 0.0005 \
--gradient_accumulation_steps 2 \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 8 \
--predict_with_generate \
--push_to_hub \
--generation_max_length 150 \
--dataloader_pin_memory False \
--disable_tqdm False \
--source_prefix 'summarize: ' \
--overwrite_output_dir \
--dataset_revision c8d80f3b036f297c713e4e2ce86b123ccc9ae961 \
--do_train \
--do_eval \
--evaluation_strategy steps \
--save_strategy steps \
--eval_steps 10000 \
--save_total_limit 1 \
--max_steps 320000 \
--logging_steps 1000 \
--save_steps 10000 \
--load_best_model_at_end True \
--metric_for_best_model eval_rougeL \
--greater_is_better True \
--optim adafactor \
--adafactor True \
--lr_scheduler_type constant \
