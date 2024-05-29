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
directory_path = '/local1/hfs/CSE481N_Project/out/hp_search/xsum'
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

# Next, graph the results by rollback/fallback. For each rb threshold,
# we plot a line with each fallback threshold
N = 6
colors = np.random.rand(N)

for idx, rb in enumerate(results.keys()):
  latencies = []
  rouges = []
  for fb in results[rb].keys():
    if fb == 0.6:
       continue
    latencies.append( 1 / results[rb][fb]['eval_samples_per_second'])
    rouges.append(results[rb][fb]['eval_rougeLsum'])
  
  areas = [100, 200, 300, 400]
  plt.plot(latencies, rouges, linestyle='-', label=f"Rollback threshold {rb}")

  plt.scatter(latencies, rouges, s=areas, c=colors[idx].resize(4), alpha=0.3)

plt.xlabel('Normalized avg latency per example')
plt.ylabel('ROUGE-L (higher better)')
plt.title('XSUM, T5')
plt.legend()
plt.savefig("xsum_tradeoff.png", dpi=300, bbox_inches='tight')