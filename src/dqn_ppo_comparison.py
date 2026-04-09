import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd

def extract_scalar(log_dir, tag):
    if not os.path.exists(log_dir):
        print(f"Warning: Directory not found: {log_dir}")
        return None
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

models = {
    'DQN Baseline': 'src/logs/keydqn_flat/dqn_final_1/',
    'DQN Curriculum': 'src/logs/keydqn_cur/S2_DoorKey5x5_1/',
    'PPO Baseline': 'src/logs/keyppo_flat/PPO_Flat_DoorKey5x5_1/',
    'PPO Curriculum': 'src/curriculum_logs/Phase2_DoorKey_0/',
    'RPPO Baseline': 'src/logs/rppo_baseline/RecurrentPPO_1/',
    'RPPO Curriculum': 'src/prelim/reppo/logs/rppo_curriculum/RecurrentPPO_0/'
}

metrics = {
    'rollout/ep_len_mean': 'Episode Length (DQN vs PPO vs RPPO).png',
    'rollout/ep_rew_mean': 'Episode Reward (DQN vs PPO vs RPPO).png'
}

titles = {
    'rollout/ep_len_mean': 'Comparison: Mean Episode Length (DQN vs PPO vs RPPO)',
    'rollout/ep_rew_mean': 'Comparison: Mean Episode Reward (DQN vs PPO vs RPPO)'
}

for metric, save_name in metrics.items():
    plt.figure(figsize=(10, 6))
    for label, log_dir in models.items():
        df = extract_scalar(log_dir, metric)
        if df is not None:
            plt.plot(df['step'], df['value'], label=label, alpha=0.8)
    
    plt.title(titles[metric])
    plt.xlabel('Timesteps')
    plt.ylabel(metric.split('/')[-1])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_name)
    print(f"Saved {save_name}")
