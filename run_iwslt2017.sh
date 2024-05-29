#!/bin/bash

model=${1:-vanilla}
fallback_threshold=${2:-0.5}
rollback_threshold=${3:-5}
max_eval_samples=${4:-888}  # debug

case "$model" in
    aligned | unaligned)
        model_name_small="kssteven/mT5-small-iwslt2017-de-en-bild"
        if [ "$model" == "aligned" ]; then
            model_name_small="kssteven/mT5-small-iwslt2017-de-en-bild-aligned"
        fi
        model_args="--model_name_large kssteven/mT5-large-iwslt2017-de-en --model_name_small $model_name_small \
        --fallback_threshold $fallback_threshold --rollback_threshold $rollback_threshold \
        --max_eval_samples $max_eval_samples"
        ;;
    vanilla)
        model_args="--model_name kssteven/mT5-large-iwslt2017-de-en"
        ;;
    *)
        echo "Invalid argument. Please use 'aligned', 'unaligned', or 'vanilla'."
        exit 1
        ;;
esac

python src/run_translation.py $model_args \
    --output_dir ./out/$model \
    --tokenizer_name google-t5/t5-small \
    --dataset_name iwslt2017 --dataset_config_name iwslt2017-de-en \
    --source_lang de --target_lang en \
    --num_beam 1 \
    --evaluation_strategy epoch --save_strategy epoch \
    --do_eval \
    --per_device_eval_batch_size 1 \
    --trust_remote_code True \
    --predict_with_generate