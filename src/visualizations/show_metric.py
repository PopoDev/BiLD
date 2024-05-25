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

    metric_keys, _ = get_metric(experiment)
    file_pattern = r'fb=(\d+(\.\d+)?)+_rb=(\d+(\.\d+)?)+\.json'
    if best == "score":
        compare_best = lambda x, y: x[metric_keys[0]] > y[metric_keys[0]]
    else:
        compare_best = lambda x, y: x[SPEED_METRIC] > y[SPEED_METRIC]

    data = []
    data_best = {}
    for file in os.listdir(results_dir):
        if experiment in file:
            with open(results_dir + file) as f:
                result = json.load(f)
                if metric_keys[0] not in result or metric_keys[1] not in result:
                    continue

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
        return ("eval_sacrebleu", "eval_meteor"), ("BLEU", "METEOR")
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
    metric_keys, metric_labels = get_metric(experiment)
    measure_func, measure_label = get_measure(speedup)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    plt.title(f"{TITLES[experiment]} {measure_label}")
    plt.xlabel(f"{measure_label}")

    ax1.set_ylabel(metric_labels[0])
    ax1.axhline(y=vanilla_data[metric_keys[0]], color='blue', linestyle='--')

    lns1 = ax1.plot(
        [measure_func(vanilla_data[SPEED_METRIC], data[SPEED_METRIC]) for data in aligned_data],
        [data[metric_keys[0]] for data in aligned_data],
        label=f"Aligned {metric_labels[0]}", marker='o', color='blue'
    )

    # lns1 = ax1.plot(
    #     [measure_func(vanilla_data[SPEED_METRIC], data[SPEED_METRIC]) for data in unaligned_data],
    #     [data[metric_keys[0]] for data in unaligned_data],
    #     label=f"Unaligned {metric_labels[0]}", marker='s', color='blue'
    # )

    # Create a second y-axis for METEOR
    ax2 = ax1.twinx()
    ax2.set_ylabel(metric_labels[1])
    ax2.axhline(y=vanilla_data[metric_keys[1]], color='red', linestyle='--')

    lns2 = ax2.plot(
        [measure_func(vanilla_data[SPEED_METRIC], data[SPEED_METRIC]) for data in aligned_data],
        [data[metric_keys[1]] for data in aligned_data],
        label=f"Aligned {metric_labels[1]}", marker='o', linestyle="--", color='red'
    )

    # lns2 = ax2.plot(
    #     [measure_func(vanilla_data[SPEED_METRIC], data[SPEED_METRIC]) for data in unaligned_data],
    #     [data[metric_keys[1]] for data in unaligned_data],
    #     label=f"Unaligned {metric_labels[1]}", marker='s', linestyle="--", color='red'
    # )

    # Combine handles and labels from both axes
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='lower right')

    plt.tight_layout()
    plt.legend()
    plt.savefig(RESULTS_DIR + f"{experiment}_{hardware}_metric_{'best_' + best if best else 'all'}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="iwslt2017", choices=["iwslt2017", "wmt14", "xsum", "cnn_dailymail"], help="Experiment to run.")
    args = parser.parse_args()

    show_performance(args.experiment, "rtx-2080", speedup=False)
    show_performance(args.experiment, "rtx-2080", speedup=False, best="score")
    show_performance(args.experiment, "rtx-2080", speedup=False, best="latency")
