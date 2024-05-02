
from argparse import ArgumentParser
import json 
import matplotlib.pyplot as plt

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
      metric = entry[args.metric]
      val_epochs.append(epoch)
      val_metrics.append(metric)

  plt.title(f'Validation {args.metric} over Epochs')
  plt.xlabel('Epochs')
  plt.ylabel(f'Validation {args.metric}')
  plt.plot(val_epochs, val_metrics, color='red')
  plt.savefig(f"{args.out_dir}/{args.dataset_name}_{args.metric}_curve.png")

  