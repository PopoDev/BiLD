#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python src/hyperparameter_search.py \
--experiment iwslt2017 \
--fb_thresholds 0.5 0.6 0.7 0.8 0.9 \
--rb_thresholds 1 2 3 4 5 6 7 8 9 10 \
--output_dir /tmp/iwslt2017_search \
--model_name_large kssteven/mT5-large-iwslt2017-de-en \
--model_name_small /tmp/finetuned_models/iwslt_aligned_smallT5_cont0/checkpoint-490000 \
--max_eval_samples 888 \
--tokenizer_name google/mt5-small \
--dataset_name iwslt2017 \
--dataset_config_name iwslt2017-de-en \
--source_lang de \
--target_lang en \