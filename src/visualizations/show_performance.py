import os
import re
import json
import argparse
import matplotlib.pyplot as plt

SPEED_METRIC = 'eval_samples_per_second'
RESULTS_DIR = 'results/'
TITLES = {
    "iwslt2017": "IWSLT 2017",
    "wmt14": "WMT 2014",
    "xsum": "XSUM",
    "cnn_dailymail": "CNN/DailyMail"
}

def calculate_speedup(baseline_runtime, new_runtime):
    return baseline_runtime / new_runtime

def calculate_latency(baseline_runtime, new_runtime):
    return new_runtime / baseline_runtime

def load_data(experiment, hardware="tesla-t4", best="latency"):
    if hardware == "tesla-t4":
        results_dir = RESULTS_DIR + "Tesla_T4-1/"
    elif hardware == "rtx-2080":
        results_dir = RESULTS_DIR + "NVIDIA_GeForce_RTX_2080_Ti-1/"
    else:
        raise ValueError(f"No results for hardware: {hardware}")

    with open(results_dir + f"vanilla/{experiment}.json") as f:
        vanilla_data = json.load(f)
    
    unaligned_data = get_data(experiment, results_dir, aligned=False, best=best)
    aligned_data = get_data(experiment, results_dir, aligned=True, best=best)

    unaligned_data = sorted(unaligned_data, key=lambda x: x[SPEED_METRIC])
    aligned_data = sorted(aligned_data, key=lambda x: x[SPEED_METRIC])

    return vanilla_data, aligned_data, unaligned_data

def get_data(experiment, results_dir, aligned, best=None):
    if aligned:
        results_dir += "aligned/"
    else:
        results_dir += "unaligned/"

    file_pattern = r'fb=(\d+(\.\d+)?)+_rb=(\d+(\.\d+)?)+\.json'
    if best == "score":
        metric_key, _ = get_metric(experiment)
        compare_best = lambda x, y: x[metric_key] > y[metric_key]
    else:
        compare_best = lambda x, y: x[SPEED_METRIC] > y[SPEED_METRIC]

    data = []
    data_best = {}
    for file in os.listdir(results_dir):
        if experiment in file:
            with open(results_dir + file) as f:
                result = json.load(f)

                if best:
                    fb_threshold = float(re.search(file_pattern, file).group(1))
                    rb_threshold = float(re.search(file_pattern, file).group(3))
                    if fb_threshold not in data_best or compare_best(result, data_best[fb_threshold]):
                        result["fb_threshold"] = fb_threshold
                        result["rb_threshold"] = rb_threshold
                        data_best[fb_threshold] = result
                else:
                    data.append(result)
    
    return list(data_best.values()) if best else data


def get_metric(experiment):
    if experiment in ["iwslt2017", "wmt14"]:
        return "eval_bleu", "BLEU"
    elif experiment in ["xsum", "cnn_dailymail"]:
        return "eval_rougeLsum", "ROUGE-L"
    else:
        raise ValueError(f"Unknown experiment: {experiment}")
    
def get_measure(speedup):
    calculate_speedup = lambda baseline, compared: baseline / compared
    calculate_normalized = lambda baseline, compared: compared / baseline
    if speedup:
        measure_func = calculate_normalized if SPEED_METRIC == 'eval_samples_per_second' else calculate_speedup
        measure_label = "Speedup"
    else:
        measure_func = calculate_speedup if SPEED_METRIC == 'eval_samples_per_second' else calculate_latency
        measure_label = "Normalized Latency"
    
    return measure_func, measure_label

def show_performance(experiment, hardware="tesla-t4", speedup=True, best=None):
    vanilla_data, aligned_data, unaligned_data = load_data(experiment, hardware, best)
    metric_key, metric_label = get_metric(experiment)
    measure_func, measure_label = get_measure(speedup)

    plt.figure(figsize=(8, 6))
    plt.title(f"{TITLES[experiment]} {measure_label}")
    plt.xlabel(f"{measure_label}")
    plt.ylabel(metric_label)

    baseline = vanilla_data[metric_key]
    plt.axhline(y=baseline-1, color='red', linestyle='--')
    plt.axhline(y=baseline, color='red', linestyle='--')
    plt.scatter(1, baseline-1, color='red', marker='x')
    plt.scatter(1, baseline, color='red', marker='x', label="Vanilla")

    plt.plot(
        [measure_func(vanilla_data[SPEED_METRIC], data[SPEED_METRIC]) for data in aligned_data],
        [data[metric_key] for data in aligned_data],
        label="Aligned", marker='s'
    )

    plt.plot(
        [measure_func(vanilla_data[SPEED_METRIC], data[SPEED_METRIC]) for data in unaligned_data],
        [data[metric_key] for data in unaligned_data],
        label="Unaligned", marker='o'
    )

    # (FB, RB) thresholds annotation
    if best:
        for data in aligned_data + unaligned_data:
            plt.annotate(f"({data['fb_threshold']}, {int(data['rb_threshold'])})", (measure_func(vanilla_data[SPEED_METRIC], data[SPEED_METRIC]), data[metric_key]))

    plt.legend()
    plt.savefig(RESULTS_DIR + f"{experiment}_{hardware}_{'speedup' if speedup else 'latency'}_{'best_' + best if best else 'all'}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="iwslt2017", choices=["iwslt2017", "wmt14", "xsum", "cnn_dailymail"], help="Experiment to run.")
    args = parser.parse_args()

    show_performance(args.experiment, "tesla-t4", speedup=True)
    show_performance(args.experiment, "tesla-t4", speedup=False)

    show_performance(args.experiment, "tesla-t4", speedup=True, best="score")
    show_performance(args.experiment, "tesla-t4", speedup=False, best="score")

    show_performance(args.experiment, "tesla-t4", speedup=True, best="latency")
    show_performance(args.experiment, "tesla-t4", speedup=False, best="latency")
