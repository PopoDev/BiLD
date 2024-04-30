huggingface-cli login --token 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
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

python src/training/run_summarization.py \
--output_dir ./out/finetuned_models/xsum_aligned_smallT5_iter2 \
--cache_dir /local1/hfs/CSE481N_Project/cache \
--model_name_or_path /local1/hfs/CSE481N_Project/out/finetuned_models/xsum_aligned_smallT5/checkpoint-89260 \
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
--save_strategy epoch \
--do_train \
--source_prefix 'summarize: ' \
--overwrite_output_dir \
--dataset_revision c8d80f3b036f297c713e4e2ce86b123ccc9ae961 