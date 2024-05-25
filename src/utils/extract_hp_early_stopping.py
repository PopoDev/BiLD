import os 
import re
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_style("whitegrid")

# Fixing random state for reproducibility
np.random.seed(19680801)

# First extract all results
directory_path = '/local1/hfs/CSE481N_Project/out/iwslt_hp_search_early_stopping/iwslt2017'
results = {}
for directory in os.listdir(directory_path):
    if os.path.isdir(os.path.join(directory_path, directory)):
        pattern = r'[-+]?\d*\.\d+|\d+'
        matches = re.findall(pattern, directory)

        file_path = os.path.join(directory_path, directory, 'eval_results.json')
        with open(file_path, 'r') as file:
          data = json.load(file)
          if int(matches[1]) not in results:
             results[int(matches[1])] = {}
          
          results[int(matches[1])][float(matches[0])] = data

# First extract all results
directory_path = '/local1/hfs/CSE481N_Project/out/iwslt_hp_search/iwslt2017'
results_baseline = {}
for directory in os.listdir(directory_path):
    if os.path.isdir(os.path.join(directory_path, directory)):
        pattern = r'[-+]?\d*\.\d+|\d+'
        matches = re.findall(pattern, directory)

        file_path = os.path.join(directory_path, directory, 'eval_results.json')
        with open(file_path, 'r') as file:
          data = json.load(file)
          if int(matches[1]) not in results_baseline:
             results_baseline[int(matches[1])] = {}
          
          results_baseline[int(matches[1])][float(matches[0])] = data

# Next, graph the results by rollback/fallback. For each rb threshold,
# we plot a line with each fallback threshold
points_to_keep = [(0.5, 4), (0.6,3), (0.5,2), (0.6,1), (0.7,1), (0.7,2), (0.8,1)]
N = 10
colors = np.random.rand(N)
# latencies = []
# rouges = []
# for idx, rb in enumerate(results.keys()):
#   for fb in results[rb].keys():
#     # if (fb, rb) not in points_to_keep:
#     #    continue
#     latencies.append( 1 / results[rb][fb]['eval_samples_per_second'])
#     rouges.append(results[rb][fb]['eval_bleu'])

#     plt.text(1 /results[rb][fb]['eval_samples_per_second'], results[rb][fb]['eval_bleu'], f'({fb}, {rb})', fontsize=9, ha='right')

# latencies = np.array(latencies)
# rouges = np.array(rouges)
# sorted_indices = np.argsort(latencies)
# latencies = latencies[sorted_indices]
# rouges = rouges[sorted_indices]

# plt.plot(latencies, rouges, linestyle='-', label=f"With early stopping")
# plt.scatter(latencies, rouges)

latencies = []
rouges = []
for idx, rb in enumerate(results_baseline.keys()):
  for fb in results_baseline[rb].keys():
    # if (fb, rb) not in points_to_keep:
    #    continue
    latencies.append( 1 / results_baseline[rb][fb]['eval_samples_per_second'])
    rouges.append(results_baseline[rb][fb]['eval_bleu'])
    plt.text(1 /results_baseline[rb][fb]['eval_samples_per_second'], results_baseline[rb][fb]['eval_bleu'], f'({fb}, {rb})', fontsize=9, ha='right')

latencies = np.array(latencies)
rouges = np.array(rouges)
sorted_indices = np.argsort(latencies)
latencies = latencies[sorted_indices]
rouges = rouges[sorted_indices]
plt.plot(latencies, rouges, linestyle='-', label=f"No early stopping")
plt.scatter(latencies, rouges)

plt.xlabel('Normalized avg latency per example')
plt.ylabel('BLEU (higher better)')
plt.title('IWSLT, T5')
plt.legend()
plt.savefig("iwslt_early_stopping_tradeoff.png", dpi=300, bbox_inches='tight')