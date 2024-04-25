import os
import json
import matplotlib.pyplot as plt

def calculate_speedup(baseline_runtime, new_runtime):
    return baseline_runtime / new_runtime

def show_speedup(experiment):
    results_dir = 'results/'

    baseline_file = f"vanilla/{experiment}.json"
    with open(results_dir + baseline_file) as f:
        baseline_data = json.load(f)
    
    unaligned_dir = results_dir + "unaligned/"
    unaligned_data = []
    for file in os.listdir(unaligned_dir):
        if experiment in file:
            with open(unaligned_dir + file) as f:
                unaligned_data.append(json.load(f))

    aligned_dir = results_dir + "aligned/"
    aligned_data = []
    for file in os.listdir(aligned_dir):
        if experiment in file:
            with open(aligned_dir + file) as f:
                aligned_data.append(json.load(f))
    
    unaligned_data = sorted(unaligned_data, key=lambda x: x["eval_runtime"])
    aligned_data = sorted(aligned_data, key=lambda x: x["eval_runtime"])

    plt.figure(figsize=(8, 6))
    plt.title(f"{experiment} speedup")
    plt.xlabel("Speedup")
    plt.ylabel("BLEU")

    plt.plot(
        [calculate_speedup(baseline_data["eval_runtime"], data["eval_runtime"]) for data in unaligned_data],
        [data["eval_bleu"] for data in unaligned_data],
        label="unaligned", marker='o'
    )

    plt.plot(
        [calculate_speedup(baseline_data["eval_runtime"], data["eval_runtime"]) for data in aligned_data],
        [data["eval_bleu"] for data in aligned_data],
        label="aligned", marker='s'
    )

    plt.axhline(y=baseline_data["eval_bleu"], color='red', linestyle='--')
    plt.scatter(1, baseline_data["eval_bleu"], color='red', marker='x', label="baseline")

    plt.legend()
    plt.show()
    plt.savefig(f"{results_dir}/{experiment}_speedup.png")


show_speedup("iwslt2017")
