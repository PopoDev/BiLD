import subprocess
import argparse

FB_THRESHOLDS = [0.2, 0.35, 0.5, 0.65, 0.8]
RB_THRESHOLDS = [0.5, 0.75, 1, 1.25, 1.5]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", default=False, help="Debug mode.")
    parser.add_argument("--experiment", type=str, default="iwslt2017", choices=["iwslt2017", "wmt2014", "xsum", "cnndm"], help="Experiment to run.")
    parser.add_argument("--aligned", action="store_true", default=False, help="Use aligned data.")
    args = parser.parse_args()

    command = []

    if args.experiment == "iwslt2017":
        command.append(f"./run_iwslt2017_{args.type}.sh")
    elif args.experiment == "wmt2014":
        raise NotImplementedError("wmt2014 is not implemented yet.")
    elif args.experiment == "xsum":
        raise NotImplementedError("xsum is not implemented yet.")
    elif args.experiment == "cnndm":
        raise NotImplementedError("cnndm is not implemented yet.")
    else:
        raise ValueError(f"Unknown experiment: {args.experiment}")

    for fb_threshold in FB_THRESHOLDS:
        for rb_threshold in RB_THRESHOLDS:
            command.append(str(fb_threshold))
            command.append(str(rb_threshold)) 
            if args.debug:
                command.append("10")
            subprocess.run(command)
