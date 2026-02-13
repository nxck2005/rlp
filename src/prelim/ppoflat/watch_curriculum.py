import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3 import PPO
import sys
import os
import time

def watch_stage(stage_num):
    # Configuration for each stage
    stages = {
        1: {
            "model": "Phase1_Model.zip",
            "env": "MiniGrid-Empty-5x5-v0",
            "desc": "PHASE 1: Navigation (Empty-5x5)"
        },
        2: {
            "model": "Phase2_Model.zip",
            "env": "MiniGrid-DoorKey-5x5-v0",
            "desc": "PHASE 2: Key & Door (DoorKey-5x5)"
        }
    }

    if stage_num not in stages:
        print(f"Error: Invalid stage {stage_num}. Available stages: 1, 2")
        return

    config = stages[stage_num]
    print(f"--- WATCHING {config['desc']} ---")
    
    # Check if model exists
    if not os.path.exists(config["model"]):
        print(f"Error: Model file '{config['model']}' not found.")
        print("Did you run 'python train_ppo_cur.py' after the recent changes?")
        return

    # Create Environment
    try:
        env = gym.make(config["env"], render_mode="human")
        env = FlatObsWrapper(env)
    except Exception as e:
        print(f"Error creating environment: {e}")
        return

    # Load Model
    # We don't need to specify policy/env args for loading usually, 
    # but passing env avoids some specific warnings.
    model = PPO.load(config["model"].replace(".zip", ""), env=env)

    # Watch Loop
    episodes = 5
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        step = 0
        
        print(f"Episode {ep+1} starts...")
        
        while not done and not truncated:
            # Deterministic=False allows some variation, True shows "best" behavior
            action, _ = model.predict(obs, deterministic=True)
            
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            step += 1
            
            # Control speed
            time.sleep(0.1)
            
        print(f"Episode {ep+1} finished: Reward {total_reward:.2f} in {step} steps.")

    env.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python watch_curriculum.py <stage_number>")
        print("Example: python watch_curriculum.py 1")
    else:
        try:
            stage = int(sys.argv[1])
            watch_stage(stage)
        except ValueError:
            print("Error: Stage number must be an integer.")
