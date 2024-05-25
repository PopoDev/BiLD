
from argparse import ArgumentParser
import json 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('--trainer_state', type=str, default=None)
  parser.add_argument('--metric', type=str, default=None)
  parser.add_argument('--dataset_name', type=str, default=None)
  parser.add_argument('--out_dir', type=str, default=None)

  args = parser.parse_args()

  file = open(args.trainer_state)
  trainer_state = json.load(file)

  val_epochs = []
  val_metrics = []
  for entry in trainer_state['log_history']:
    if 'eval_loss' in entry:
      epoch = entry['epoch']
      metric = entry[f"eval_{args.metric}"]
      val_epochs.append(epoch)
      val_metrics.append(metric)

  # Extract the 
  align_trainer_state_dir = '/local1/hfs/CSE481N_Project/data/trainer_states/iswslt_aligned_small.json'
  val_epochs_ours = []
  val_metrics_ours = []
  epoch = 0
  for i in range(0):
    file = open(f"{align_trainer_state_dir}/trainer_state{i}.json")
    state = json.load(file)
    last_epoch = 0
    for entry in state['log_history']:
      if 'eval_loss' in entry:
        epoch += (entry['epoch'] - last_epoch)
        last_epoch = entry['epoch']
        metric = entry[f"eval_{args.metric}"]
        val_epochs_ours.append(epoch)
        val_metrics_ours.append(metric)

  plt.figure(figsize=(10, 8))  # Set figure size

  # Plot original and reproduction curves with labels and colors
  plt.plot(val_epochs, val_metrics, color='#FF5733', linewidth=2, label='Original')
  plt.plot(val_epochs_ours, val_metrics_ours, color='#33BFFF', linewidth=2, linestyle='--', label='Reproduction')

  # Set title, labels, and legend with appropriate font sizes
  plt.title(f'XSUM Alignment: {args.metric} vs Epochs', fontsize=20, fontweight='bold')
  plt.xlabel('Epochs', fontsize=18)
  plt.ylabel(f'Validation {args.metric}', fontsize=18)
  plt.legend(fontsize=16)

  # Set tick parameters for better readability
  plt.tick_params(axis='both', which='major', labelsize=14)
  plt.tick_params(axis='both', which='minor', labelsize=12)

  # Add grid lines for clarity
  plt.grid(True, linestyle='--', alpha=0.6)

  # Save the plot with specified filename and format
  plt.savefig(f"{args.out_dir}/{args.dataset_name}_{args.metric}_curve_compared.png", dpi=300, bbox_inches='tight')
  plt.show()
  