import subprocess
import argparse
from string import Template 
import torch

FB_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
RB_THRESHOLDS = [1, 2, 3, 4, 5]

command_template = Template(
"""python src/run_summarization.py 
--output_dir $output_dir
--model_name_large $model_name_large 
--model_name_small $model_name_small
--tokenizer_name $tokenizer_name
--dataset_name $dataset_name 
--source_prefix $source_prefix
--fallback_threshold $fallback_threshold 
--rollback_threshold $rollback_threshold 
--num_beam 1 
--evaluation_strategy epoch --save_strategy epoch 
--do_eval 
--per_device_eval_batch_size 1 
--trust_remote_code True 
--predict_with_generate
--max_eval_samples $max_eval_samples"""
)

def cleanup():
    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="iwslt2017", choices=["iwslt2017", "wmt2014", "xsum", "cnndm"], help="Experiment to run.")
    parser.add_argument("--fb_thresholds", nargs="+", type=float, default=FB_THRESHOLDS, help="List of FB thresholds.")
    parser.add_argument("--rb_thresholds", nargs="+", type=int, default=RB_THRESHOLDS, help="List of RB thresholds.")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory to put results.")
    parser.add_argument("--model_name_large", type=str, default=None, help="Large model")
    parser.add_argument("--model_name_small", type=str, default=None, help="Small model")
    parser.add_argument("--source_prefix", type=str, default='', help="Source prefix to use in eval.")
    parser.add_argument("--max_eval_samples", type=str, default='', help="Number of samples to use in eval.")
    parser.add_argument("--tokenizer_name", type=str, default='', help="Tokenizer name.")
    parser.add_argument("--dataset_name", type=str, default='', help="Dataset name.")

    args = parser.parse_args()

    for fb_threshold in args.fb_thresholds:
        for rb_threshold in args.rb_thresholds:
            command = command_template.substitute({
                'output_dir': f"{args.output_dir}/{args.experiment}/fb={fb_threshold}_rb={rb_threshold}",
                'model_name_large': args.model_name_large,
                'model_name_small': args.model_name_small,
                'tokenizer_name': args.tokenizer_name,
                'dataset_name': args.dataset_name,
                'source_prefix': args.source_prefix,
                'fallback_threshold': fb_threshold,
                'rollback_threshold': rb_threshold,
                'max_eval_samples': args.max_eval_samples
            })
            command = command.split()
            subprocess.run(command)
            torch.cuda.empty_cache()
    cleanup()




