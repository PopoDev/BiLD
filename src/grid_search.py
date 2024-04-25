import torch
import subprocess
import argparse
import torch.multiprocessing as mp

FB_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
RB_THRESHOLDS = [1, 2, 3, 4, 5]

def cleanup():
    torch.distributed.destroy_process_group()

def run(rank, world_size, args):
    if args.experiment == "iwslt2017":
        script_name = f"./run_iwslt2017_{'aligned' if args.aligned else 'unaligned'}.sh"
    elif args.experiment == "wmt2014":
        script_name = f"./run_wmt2014_{'aligned' if args.aligned else 'unaligned'}.sh"
    elif args.experiment == "xsum":
        raise NotImplementedError("xsum is not implemented yet.")
    elif args.experiment == "cnndm":
        raise NotImplementedError("cnndm is not implemented yet.")
    else:
        raise ValueError(f"Unknown experiment: {args.experiment}")

    fb_thresholds = FB_THRESHOLDS[rank * len(FB_THRESHOLDS) // world_size : (rank + 1) * len(FB_THRESHOLDS) // world_size]

    for fb_threshold in fb_thresholds:
        if args.aligned:
            fb_threshold += 0.1

        for rb_threshold in RB_THRESHOLDS:
            command = [script_name, str(fb_threshold), str(rb_threshold)]
            if args.debug:
                command.append("10")
            subprocess.run(command)

    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", default=False, help="Debug mode.")
    parser.add_argument("--experiment", type=str, default="iwslt2017", choices=["iwslt2017", "wmt2014", "xsum", "cnndm"], help="Experiment to run.")
    parser.add_argument("--aligned", action="store_true", default=False, help="Use aligned data.")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs.")
    args = parser.parse_args()

    world_size = args.gpus
    mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)
