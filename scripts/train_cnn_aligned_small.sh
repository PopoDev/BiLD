huggingface-cli login --token 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
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false

python src/training/run_summarization.py \
--output_dir /tmp/finetuned_models/cnn_aligned_smallT5_cont3 \
--cache_dir /tmp/cache \
--model_name_or_path /tmp/finetuned_models/cnn_aligned_smallT5_cont2/checkpoint-40000 \
--tokenizer_name google-t5/t5-small \
--dataset_name lilferrit/cnn_dailymail_t5_distillation \
--text_column article \
--summary_column t5_large_output \
--learning_rate 0.0002 \
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
--dataset_revision 7f69fe7b38ad72f82bfdc31f00d89866e751959a \
--do_train \
--do_eval \
--evaluation_strategy steps \
--save_strategy steps \
--eval_steps 10000 \
--save_total_limit 1 \
--max_steps 380000 \
--logging_steps 1000 \
--save_steps 10000 \
--load_best_model_at_end True \
--metric_for_best_model eval_rougeL \
--greater_is_better True \
--optim adafactor \
--adafactor True \
--lr_scheduler_type constant \