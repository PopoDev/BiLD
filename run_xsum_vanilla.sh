#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

python src/run_summarization.py \
    --output_dir /local1/hfs/gs_stuff/vn-xsum-bild-res \
    --model_name kssteven/T5-large-xsum \
    --tokenizer_name google-t5/t5-small \
    --dataset_name xsum \
    --source_prefix "summarize: " \
    --num_beam 1 \
    --evaluation_strategy epoch --save_strategy epoch \
    --do_eval \
    --per_device_eval_batch_size 1 \
    --trust_remote_code True \
    --predict_with_generate
