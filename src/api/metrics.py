import os
import glob
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# --- CONFIGURATION ---
# Default smoothing factor (0.0 = no smoothing, 0.99 = very smooth)
SMOOTHING_FACTOR = 0.85
MAX_POINTS = 200

def extract_metrics(log_dir, max_points=MAX_POINTS, smoothing=SMOOTHING_FACTOR):
    """
    Recursively finds ALL event files in log_dir, aggregates data, 
    applies smoothing (EMA), and returns a step-sorted list.
    """
    event_files = glob.glob(os.path.join(log_dir, "**", "events.out.tfevents.*"), recursive=True)
        
    if not event_files:
        return {"error": f"No event files found in {log_dir}"}

    mapping = {
        "reward": ["rollout/ep_rew_mean", "train/reward", "main/reward", "reward"],
        "success_rate": ["rollout/success_rate", "main/success_rate", "train/success_rate", "success_rate"],
        "loss": ["train/loss", "train/value_loss", "train/explained_variance", "loss"],
        "entropy": ["train/entropy_loss", "rollout/entropy", "entropy"]
    }

    results = {k: [] for k in mapping.keys()}
    
    for event_file in event_files:
        try:
            acc = EventAccumulator(event_file, size_guidance={'scalars': 0})
            acc.Reload()
            tags = acc.Tags().get('scalars', [])
            
            for key, possible_tags in mapping.items():
                found_tag = next((t for t in possible_tags if t in tags), None)
                if found_tag:
                    events = acc.Scalars(found_tag)
                    for e in events:
                        results[key].append({"step": e.step, "value": float(e.value)})
        except Exception as e:
            print(f"Error reading {event_file}: {e}")

    final_results = {}
    for key, data in results.items():
        if not data:
            continue
            
        data.sort(key=lambda x: x["step"])
        
        # Deduplicate steps
        unique_data = []
        if data:
            unique_data.append(data[0])
            for i in range(1, len(data)):
                if data[i]["step"] == unique_data[-1]["step"]:
                    unique_data[-1]["value"] = data[i]["value"]
                else:
                    unique_data.append(data[i])
        
        # Apply Smoothing (Exponential Moving Average)
        if smoothing > 0 and len(unique_data) > 0:
            smoothed = []
            last = unique_data[0]["value"]
            for point in unique_data:
                # Formula: smoothed = weight * last + (1 - weight) * current
                de_noised = last * smoothing + (1 - smoothing) * point["value"]
                smoothed.append({"step": point["step"], "value": de_noised})
                last = de_noised
            unique_data = smoothed

        # Downsample
        if len(unique_data) > max_points:
            indices = np.linspace(0, len(unique_data) - 1, max_points).astype(int)
            unique_data = [unique_data[i] for i in indices]
            
        final_results[key] = unique_data
            
    return final_results

def get_log_dir_for_model(model_id, base_dir, models_dir):
    """
    Finds the log directory associated with a specific model ID.
    """
    log_map = {
        "dqn_baseline": os.path.join(base_dir, "logs/keydqn_flat"),
        "dqn_framestack": os.path.join(base_dir, "logs/keydqn_framestack"),
        "dqn_custom_cnn": os.path.join(base_dir, "prelim/qlearning/logs"),
        "dqn_curriculum": os.path.join(base_dir, "logs/keydqn_cur"),
        
        "ppo_flat": os.path.join(base_dir, "logs/keyppo_flat"),
        "ppo_flat_cur": os.path.join(base_dir, "curriculum_logs"),
        "ppo_baseline": os.path.join(base_dir, "logs/Baseline_NoCurriculum_1"),
        "ppo_curriculum": os.path.join(base_dir, "logs/CurrS3Final_DoorKey8x8_1"),
        
        "rppo_baseline": os.path.join(base_dir, "logs/rppo_baseline"),
        "rppo_curriculum": os.path.join(base_dir, "prelim/reppo/logs/rppo_curriculum"),
        
        "repp2_4stage": os.path.join(base_dir, "prelim/repp2/logs/rppo4Stage"),
    }
    
    return log_map.get(model_id)
