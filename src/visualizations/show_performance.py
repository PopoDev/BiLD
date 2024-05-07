import os
import re
import json
import matplotlib.pyplot as plt

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

def load_data(experiment, hardware="tesla-t4", best=False):
    if hardware == "tesla-t4":
        results_dir = RESULTS_DIR + "Tesla_T4-1/"
    elif hardware == "rtx-2080":
        results_dir = RESULTS_DIR + "NVIDIA_GeForce_RTX_2080_Ti-1/"
    else:
        raise ValueError(f"No results for hardware: {hardware}")

    file_pattern = r'fb=(\d+\.\d+)_rb=(\d+\.\d+)\.json'
    with open(results_dir + f"vanilla/{experiment}.json") as f:
        vanilla_data = json.load(f)
    
    unaligned_data = []
    unaligned_data_best = {}
    for file in os.listdir(results_dir + "unaligned/"):
        if experiment in file:
            with open(results_dir + "unaligned/" + file) as f:
                data = json.load(f)

                if best:
                    fb_threshold = float(re.search(file_pattern, file).group(1))
                    rb_threshold = float(re.search(file_pattern, file).group(2))
                    if fb_threshold not in unaligned_data_best or data["eval_runtime"] < unaligned_data_best[fb_threshold]["eval_runtime"]:
                        data["fb_threshold"] = fb_threshold
                        data["rb_threshold"] = rb_threshold
                        unaligned_data_best[fb_threshold] = data
                else:
                    unaligned_data.append(data)

    aligned_data = []
    aligned_data_best = {}
    for file in os.listdir(results_dir + "aligned/"):
        if experiment in file:
            with open(results_dir + "aligned/" + file) as f:
                data = json.load(f)

                if best:
                    fb_threshold = float(re.search(file_pattern, file).group(1))
                    rb_threshold = float(re.search(file_pattern, file).group(2))
                    if fb_threshold not in aligned_data_best or data["eval_runtime"] < aligned_data_best[fb_threshold]["eval_runtime"]:
                        data["fb_threshold"] = fb_threshold
                        data["rb_threshold"] = rb_threshold
                        aligned_data_best[fb_threshold] = data
                else:
                    aligned_data.append(data)
    
    if best:
        unaligned_data = list(unaligned_data_best.values())
        aligned_data = list(aligned_data_best.values())

    unaligned_data = sorted(unaligned_data, key=lambda x: x["eval_runtime"])
    aligned_data = sorted(aligned_data, key=lambda x: x["eval_runtime"])

    return vanilla_data, aligned_data, unaligned_data

def get_metric(experiment):
    if experiment in ["iwslt2017", "wmt14"]:
        return "eval_bleu", "BLEU"
    elif experiment in ["xsum", "cnndm"]:
        return "eval_rougeLsum", "ROUGE-L"
    
def get_measure(speedup):
    if speedup:
        return calculate_speedup, "Speedup"
    else:
        return calculate_latency, "Normalized Latency"

def show_performance(experiment, hardware="tesla-t4", speedup=True, best=False):
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
        [measure_func(vanilla_data["eval_runtime"], data["eval_runtime"]) for data in aligned_data],
        [data[metric_key] for data in aligned_data],
        label="Aligned", marker='s'
    )

    plt.plot(
        [measure_func(vanilla_data["eval_runtime"], data["eval_runtime"]) for data in unaligned_data],
        [data[metric_key] for data in unaligned_data],
        label="Unaligned", marker='o'
    )

    # (RB, FB) thresholds annotation
    if best:
        for data in aligned_data + unaligned_data:
            plt.annotate(f"({int(data['rb_threshold'])}, {data['fb_threshold']})", (measure_func(vanilla_data["eval_runtime"], data["eval_runtime"]), data[metric_key]))

    plt.legend()
    plt.show()
    plt.savefig(RESULTS_DIR + f"{experiment}_{hardware}_{'speedup' if speedup else 'latency'}_{'best' if best else 'all'}.png")

if __name__ == "__main__":
    show_performance("wmt14", "tesla-t4", speedup=True)
    show_performance("wmt14", "tesla-t4", speedup=False)

    show_performance("wmt14", "tesla-t4", speedup=True, best=True)
    show_performance("wmt14", "tesla-t4", speedup=False, best=True)
