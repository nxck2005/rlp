import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd

def extract_scalar(log_dir, tag):
    event_files = sorted([f for f in os.listdir(log_dir) if 'events.out.tfevents' in f])
    all_data = []
    for event_file in event_files:
        path = os.path.join(log_dir, event_file)
        ea = event_accumulator.EventAccumulator(path)
        ea.Reload()
        if tag in ea.Tags()['scalars']:
            data = ea.Scalars(tag)
            all_data.append(pd.DataFrame(data))
    if not all_data: return None
    return pd.concat(all_data).sort_values('step').drop_duplicates('step')

log_dir = 'src/prelim/repp2/logs/rppo4Stage/RecurrentPPO_0/'
tag = 'rollout/ep_rew_mean'

print(f"Extracting RPPO 4-Stage data from {log_dir}...")
df = extract_scalar(log_dir, tag)

if df is not None:
    plt.figure(figsize=(10, 6))
    plt.plot(df['step'], df['value'], color='green', label='RPPO 4-Stage')
    plt.title('RPPO training performance')
    plt.xlabel('Timesteps')
    plt.ylabel('Mean Episode Reward')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    save_path = 'rppo_4stage_performance.png'
    plt.savefig(save_path)
    print(f"  Steps: {df['step'].min()} to {df['step'].max()} (count: {len(df)})")
    print(f"Plot saved to {save_path}")
else:
    print("Could not find data for the specified tag.")
