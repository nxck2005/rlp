import gymnasium as gym
import minigrid
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
from stable_baselines3 import DQN
import os
import sys
import time
import torch as th

# Directories - relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# New models directory: src/models/keydqn
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR))), "src", "models", "keydqn")
# New logs directory: src/logs/keydqn
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR))), "src", "logs", "keydqn")

# Environment ID
ENV_ID = "MiniGrid-DoorKey-5x5-v0"
# Model Name
MODEL_NAME = "DQN_Pixels_DoorKey5x5_Baseline"

def train():
    print(f"--- STARTING DQN PIXEL BASELINE TRAINING ({ENV_ID}) ---")
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # 1. Create the environment
    env = gym.make(ENV_ID, render_mode="rgb_array")
    
    # 2. Wrap it to output PIXELS (56x56x3)
    env = RGBImgPartialObsWrapper(env, tile_size=8)
    env = ImgObsWrapper(env)

    print(f"Observation Space: {env.observation_space.shape}")

    # 3. Standard DQN with Standard CnnPolicy
    model = DQN(
        "CnnPolicy",
        env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        device="cuda",
        buffer_size=100000,
        learning_rate=1e-4,
        exploration_fraction=0.2, # Slightly higher exploration for key/door
        exploration_final_eps=0.05,
        target_update_interval=1000,
        train_freq=4,
        gradient_steps=1,
    )

    # 4. Train
    print("Training for 500,000 steps...")
    model.learn(total_timesteps=500000, tb_log_name=MODEL_NAME)
    
    # 5. Save
    save_path = os.path.join(MODELS_DIR, MODEL_NAME)
    model.save(save_path)
    print(f"Model saved to {save_path}")
    env.close()

def watch(max_steps_per_episode=None):
    print(f"--- WATCHING DQN PIXEL AGENT ({ENV_ID}) ---")
    
    model_path = os.path.join(MODELS_DIR, MODEL_NAME)
    if not os.path.exists(model_path + ".zip"):
        print(f"Model not found at {model_path}.zip")
        return

    try:
        env = gym.make(ENV_ID, render_mode="human")
    except Exception as e:
        print(f"Error creating env: {e}")
        return
    
    # Apply same wrappers as training
    env = RGBImgPartialObsWrapper(env, tile_size=8)
    env = ImgObsWrapper(env)

    model = DQN.load(model_path, env=env)

    for ep in range(5):
        obs, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        step_count = 0
        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            step_count += 1
            if max_steps_per_episode is not None and step_count >= max_steps_per_episode:
                truncated = True 
            time.sleep(0.05)
        print(f"Episode {ep+1}: Reward {total_reward:.4f} (Steps: {step_count})")
    
    env.close()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--watch":
        max_steps = int(sys.argv[2]) if len(sys.argv) > 2 else None
        watch(max_steps)
    else:
        train()
