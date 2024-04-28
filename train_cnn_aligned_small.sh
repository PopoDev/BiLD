huggingface-cli login --token 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
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false

python src/training/run_summarization.py \
--output_dir ./out/finetuned_models/cnn_aligned_smallT5_iter1 \
--cache_dir /local1/hfs/CSE481N_Project/cache \
--model_name_or_path google-t5/t5-small \
--tokenizer_name google-t5/t5-small \
--dataset_name lilferrit/cnn_dailymail_t5_distillation \
--text_column article \
--summary_column t5_large_output \
--learning_rate 0.0005 \
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
--dataset_revision 7f69fe7b38ad72f82bfdc31f00d89866e751959a \