python src/run_translation.py \
--output_dir ./out/t5bild/unaligned \
--model_name_large kssteven/mT5-large-iwslt2017-de-en --model_name_small kssteven/mT5-small-iwslt2017-de-en \
--tokenizer_name google/mt5-small \
--dataset_name iwslt2017 --dataset_config_name iwslt2017-de-en \
--source_lang de --target_lang en \
--fallback_threshold 0.5 --rollback_threshold 5 \
--metric_for_best_model bleu \
--num_beam 1 \
--evaluation_strategy epoch --save_strategy epoch \
--do_eval \
<<<<<<< HEAD
--max_eval_samples 202 \
=======
>>>>>>> e74e68873a1f157ead94f684b237f08dcad72bc7
--per_device_eval_batch_size 1 \
--predict_with_generate
