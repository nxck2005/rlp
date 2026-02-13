import gymnasium as gym
import minigrid
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import DQN
import os
import sys
import time
from .minigrid_cnn import MinigridCNN

# --- MinigridCNN class definition removed ---

def watch_dqn(model_path, env_id, delay=0.1, n_episodes=5):
    print(f"Loading DQN model from: {model_path}")
    print(f"Environment: {env_id}")
    
    # 1. Setup Environment with 'human' render mode
    try:
        env = gym.make(env_id, render_mode="human")
    except Exception as e:
        print(f"Error creating env: {e}")
        return

    env = ImgObsWrapper(env)
    
    # 2. Load the Agent with custom policy kwargs
    if not os.path.exists(model_path) and not os.path.exists(model_path + ".zip"):
        print(f"Error: Model file not found at {model_path}.zip")
        return

    model = DQN.load(
        model_path, 
        env=env, 
        device="cuda",
        # Important: policy_kwargs must match how the model was trained
        policy_kwargs=dict(
            features_extractor_class=MinigridCNN,
            features_extractor_kwargs=dict(features_dim=128),
        ),
    )

    # 3. Play Loop
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        step = 0
        
        print(f"--- Episode {ep+1} ---")
        
        while not done and not truncated:
            action, _states = model.predict(obs, deterministic=True) # deterministic=True for watching
            
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            
            time.sleep(delay)
            
        print(f"Finished in {step} steps. Reward: {total_reward:.4f}")

    env.close()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python watch_dqn.py <model_path> <env_id> [delay_s] [num_episodes]")
        print("Example: python watch_dqn.py models/dqn_curriculum/S1_weights MiniGrid-Empty-8x8-v0")
        sys.exit(1)
    
    model_path = sys.argv[1]
    env_id = sys.argv[2]
    delay = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
    n_episodes = int(sys.argv[4]) if len(sys.argv) > 4 else 5

    watch_dqn(model_path, env_id, delay, n_episodes)
