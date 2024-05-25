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
directory_path = '/local1/hfs/CSE481N_Project/results/NVIDIA_GeForce_RTX_2080_Ti-1/aligned'
results = {}
for directory in os.listdir(directory_path):
    if 'iwslt' not in directory:
       continue
    if os.path.isfile(os.path.join(directory_path, directory)):
        pattern = r'(?<=fb=|rb=)(\d*\.\d+)'
        matches = re.findall(pattern, directory)
        file_path = os.path.join(directory_path, directory)
        with open(file_path, 'r') as file:
          data = json.load(file)
          if float(matches[1]) not in results:
            results[float(matches[1])] = {}
          
          results[float(matches[1])][float(matches[0])] = data

# Next, graph the results by rollback/fallback. For each rb threshold,
# we plot a line with each fallback threshold
N = 6
colors = np.random.rand(N)
for idx, rb in enumerate(results.keys()):
  latencies = []
  rouges = []
  for fb in results[rb].keys():

    latencies.append( 1 / results[rb][fb]['eval_samples_per_second'])
    rouges.append(results[rb][fb]['eval_bleu'])
  
  areas = [100, 150, 200, 250, 300, 350, 400, 450, 500]
  plt.plot(latencies, rouges, linestyle='-', label=f"Rollback threshold {rb}")
  num_points = len(results[rb].keys())
  plt.scatter(latencies, rouges, s=areas[0:num_points], c=colors[idx].resize(num_points), alpha=0.3)

plt.xlabel('Normalized avg latency per example')
plt.ylabel('BLEU (higher better)')
plt.title('IWSLT, T5')
plt.legend()
plt.savefig("iwslt_tradeoff.png", dpi=300, bbox_inches='tight')