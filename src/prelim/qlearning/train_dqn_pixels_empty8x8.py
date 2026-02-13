import gymnasium as gym
import minigrid
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
from stable_baselines3 import DQN
import os
import sys
import time
import torch as th

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models", "dqn_pixels")
LOG_DIR = os.path.join(BASE_DIR, "logs")

def train():
    print("--- STARTING DQN PIXEL TRAINING (Standard CnnPolicy) ---")
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    env = gym.make("MiniGrid-Empty-8x8-v0", render_mode="rgb_array")
    env = RGBImgPartialObsWrapper(env, tile_size=8)
    env = ImgObsWrapper(env)

    print(f"Observation Space: {env.observation_space.shape}")

    model = DQN(
        "CnnPolicy",
        env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        device="cuda",
        buffer_size=100000,
        learning_rate=1e-3,
        exploration_fraction=0.1,
        exploration_final_eps=0.05,
        target_update_interval=1000,
        train_freq=4,
        gradient_steps=1,
    )

    print("Training for 500,000 steps...")
    model.learn(total_timesteps=500000, tb_log_name="DQN_Pixels_Empty8x8")

    save_path = os.path.join(MODELS_DIR, "Empty8x8_Pixel_DQN")
    model.save(save_path)
    print(f"Model saved to {save_path}")
    env.close()

def watch(max_steps_per_episode=None):
    print("--- WATCHING DQN PIXEL AGENT ---")
    
    model_path = os.path.join(MODELS_DIR, "Empty8x8_Pixel_DQN")
    if not os.path.exists(model_path + ".zip"):
        print(f"Model not found at {model_path}.zip")
        return

    try:
        env = gym.make("MiniGrid-Empty-8x8-v0", render_mode="human")
    except Exception as e:
        print(f"Error creating env: {e}")
        return
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
                truncated = True # Force truncation if step limit reached
            time.sleep(0.05)
        print(f"Episode {ep+1}: Reward {total_reward:.4f} (Steps: {step_count})")
    
    env.close()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--watch":
        max_steps = int(sys.argv[2]) if len(sys.argv) > 2 else None
        watch(max_steps)
    else:
        train()

