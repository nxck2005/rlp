import gymnasium as gym
import minigrid
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
import sys
import os
import time

def watch(model_path, env_id="MiniGrid-DoorKey-5x5-v0", delay=0.1):
    print(f"Loading model from: {model_path}")
    print(f"Environment: {env_id}")
    
    # 1. Setup Environment with 'human' render mode
    try:
        env = gym.make(env_id, render_mode="human")
    except Exception as e:
        print(f"Error creating env: {e}")
        return

    env = ImgObsWrapper(env)
    
    # 2. Load the Agent
    if not os.path.exists(model_path + ".zip") and not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    model = PPO.load(model_path)

    # 3. Play Loop
    episodes = 5
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        step = 0
        
        print(f"--- Episode {ep+1} ---")
        
        while not done and not truncated:
            # Predict action
            action, _states = model.predict(obs, deterministic=False) # False = let it explore a bit, True = best move only
            
            # Take step
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            
            # Slow down so human can see
            time.sleep(delay)
            
        print(f"Finished in {step} steps. Reward: {total_reward:.4f}")

    env.close()

if __name__ == "__main__":
    # Default values
    default_model = "src/prelim/models/baseline/baseline_model"
    default_env = "MiniGrid-DoorKey-5x5-v0"
    
    if len(sys.argv) > 1:
        m_path = sys.argv[1]
    else:
        m_path = default_model

    if len(sys.argv) > 2:
        e_id = sys.argv[2]
    else:
        e_id = default_env

    watch(m_path, e_id)
