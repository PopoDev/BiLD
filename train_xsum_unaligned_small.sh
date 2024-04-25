huggingface-cli login --token $HUGGINGFACE_TOKEN

CUDA_VISIBLE_DEVICES=0 python src/run_training.py \
--output_dir ./out/finetuned_models/xsum_unaligned_smallT5 \
--cache_dir /local1/hfs/CSE481N_Project/cache \
--model_name google-t5/t5-small \
--tokenizer_name google-t5/t5-small \
--dataset_name EdinburghNLP/xsum \
--metric rouge \
--learning_rate 0.0002 \
--gradient_accumulation_steps 2 \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 1 \
--save_total_limit 3 \
--max_steps 500000 \
--predict_with_generate \
--push_to_hub \
--generation_max_length 100 \
--evaluation_strategy steps \
--eval_steps 2000