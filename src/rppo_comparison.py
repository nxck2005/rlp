import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd

def extract_scalar(log_dir, tag):
    # Get all event files in the directory
    event_files = sorted([f for f in os.listdir(log_dir) if 'events.out.tfevents' in f])
    all_data = []
    
    for event_file in event_files:
        path = os.path.join(log_dir, event_file)
        ea = event_accumulator.EventAccumulator(path)
        ea.Reload()
        
        if tag in ea.Tags()['scalars']:
            data = ea.Scalars(tag)
            df = pd.DataFrame(data)
            all_data.append(df)
            
    if not all_data:
        return None
        
    # Concatenate data and sort by step (important if multiple files exist)
    df = pd.concat(all_data).sort_values('step').drop_duplicates('step')
    return df

# Directories
baseline_dir = 'src/logs/rppo_baseline/RecurrentPPO_1/'
curriculum_dir = 'src/prelim/reppo/logs/rppo_curriculum/RecurrentPPO_0/'
tag = 'rollout/ep_rew_mean'

# Extract
print(f"Extracting baseline data from {baseline_dir}...")
df_baseline = extract_scalar(baseline_dir, tag)
if df_baseline is not None:
    print(f"  Baseline steps: {df_baseline['step'].min()} to {df_baseline['step'].max()} (count: {len(df_baseline)})")

print(f"Extracting curriculum data from {curriculum_dir}...")
df_curriculum = extract_scalar(curriculum_dir, tag)
if df_curriculum is not None:
    print(f"  Curriculum steps: {df_curriculum['step'].min()} to {df_curriculum['step'].max()} (count: {len(df_curriculum)})")

# Plotting
plt.figure(figsize=(10, 6))

if df_baseline is not None:
    plt.plot(df_baseline['step'], df_baseline['value'], label='RPPO Baseline', alpha=0.8)
    
if df_curriculum is not None:
    plt.plot(df_curriculum['step'], df_curriculum['value'], label='RPPO Curriculum', alpha=0.8)

plt.title('Comparison: RPPO Baseline vs RPPO Curriculum (8x8)')
plt.xlabel('Timesteps')
plt.ylabel('Mean Episode Reward')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# Save
save_path = 'rppo_comparison.png'
plt.savefig(save_path)
print(f"Plot saved to {save_path}")
