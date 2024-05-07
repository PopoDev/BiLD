import os
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

def load_data(experiment, hardware="tesla-t4"):
    if hardware == "tesla-t4":
        results_dir = RESULTS_DIR + "Tesla_T4-1/"
    elif hardware == "rtx-2080":
        results_dir = RESULTS_DIR + "NVIDIA_GeForce_RTX_2080_Ti-1/"
    else:
        raise ValueError(f"No results for hardware: {hardware}")

    with open(results_dir + f"vanilla/{experiment}.json") as f:
        vanilla_data = json.load(f)
    
    unaligned_data = []
    for file in os.listdir(results_dir + "unaligned/"):
        if experiment in file:
            with open(results_dir + "unaligned/" + file) as f:
                unaligned_data.append(json.load(f))

    aligned_data = []
    for file in os.listdir(results_dir + "aligned/"):
        if experiment in file:
            with open(results_dir + "aligned/" + file) as f:
                aligned_data.append(json.load(f))
    
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

def show_performance(experiment, hardware="tesla-t4", speedup=True):
    vanilla_data, aligned_data, unaligned_data = load_data(experiment, hardware)
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

    plt.legend()
    plt.show()
    plt.savefig(RESULTS_DIR + f"{experiment}_{hardware}_{'speedup' if speedup else 'latency'}.png")

if __name__ == "__main__":
    show_performance("wmt14", "tesla-t4", speedup=True)
    show_performance("wmt14", "tesla-t4", speedup=False)
