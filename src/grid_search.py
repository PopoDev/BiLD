import subprocess
import argparse

FB_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
RB_THRESHOLDS = [1, 2, 3, 4, 5]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", default=False, help="Debug mode.")
    parser.add_argument("--experiment", type=str, default="iwslt2017", choices=["iwslt2017", "wmt2014", "xsum", "cnndm"], help="Experiment to run.")
    parser.add_argument("--aligned", action="store_true", default=False, help="Use aligned data.")
    parser.add_argument("--gpu", type=int, help="GPU to use.")
    args = parser.parse_args()

    command = []

    if args.experiment == "iwslt2017":
        command.append(f"./run_iwslt2017_{'aligned' if args.aligned else 'unaligned'}.sh")
    elif args.experiment == "wmt2014":
        command.append(f"./run_wmt2014_{'aligned' if args.aligned else 'unaligned'}.sh")
    elif args.experiment == "xsum":
        raise NotImplementedError("xsum is not implemented yet.")
    elif args.experiment == "cnndm":
        raise NotImplementedError("cnndm is not implemented yet.")
    else:
        raise ValueError(f"Unknown experiment: {args.experiment}")
    
    if args.gpu == 0:
        fb_thresholds = FB_THRESHOLDS[:len(FB_THRESHOLDS) // 2]
    elif args.gpu == 1:
        fb_thresholds = FB_THRESHOLDS[len(FB_THRESHOLDS) // 2:]
    
    for fb_threshold in fb_thresholds:
        for rb_threshold in RB_THRESHOLDS:
            command.append(str(fb_threshold))
            command.append(str(rb_threshold)) 
            if args.debug:
                command.append("10")
            subprocess.run(command)
