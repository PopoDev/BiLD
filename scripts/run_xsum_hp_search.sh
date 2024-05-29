#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python src/hyperparameter_search.py \
--experiment xsum \
--fb_thresholds 0.2 0.3 0.4 0.5 0.6 \
--rb_thresholds 1 2 3 4 5 6 \
--output_dir /local1/hfs/CSE481N_Project/out/hp_search \
--model_name_large kssteven/T5-large-xsum \
--model_name_small /local1/hfs/CSE481N_Project/data/finetuned_models/xsum_aligned_smallT5_cont3/checkpoint-210000 \
--max_eval_samples 1000 \
--tokenizer_name google-t5/t5-small \
--dataset_name xsum \
--source_prefix "summarize: "