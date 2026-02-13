import gymnasium as gym
import minigrid
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
from stable_baselines3 import DQN
import os
import sys
import time
import torch as th

# HYPER PARAMS
EXPLORATION_FRACTION = 0.5
FINAL_EXPLORATION_DECAY = 0.1
ITERATIONS = 200_000
LEARNING_RATE = 0.0002
WATCH_EPISODES = 10
MEM_USED = 400_000
# Collect x steps and then update gradients y times.
WEIGHT_COLLECT_FREQ = 4
GRADIENT_UPDATE_FREQ = 2
START_LEARNING_AT_STEP = 5000
# great amt of vram so no problems
BATCH_SIZE = 128

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR))), "src", "models", "keydqn")
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR))), "src", "logs", "keydqn")
ENV_ID = "MiniGrid-DoorKey-5x5-v0"
MODEL_NAME = "DQN_Pixels_DoorKey5x5_Baseline"

def train():
    print(f"--- STARTING DQN PIXEL BASELINE TRAINING ({ENV_ID}) ---")
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    env = gym.make(ENV_ID, render_mode="rgb_array")
    env = RGBImgPartialObsWrapper(env, tile_size=8)
    env = ImgObsWrapper(env)

    print(f"Observation Space: {env.observation_space.shape}")

    model = DQN(
        "CnnPolicy",
        env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        device="cuda",
        buffer_size=MEM_USED,
        learning_rate=LEARNING_RATE,
        exploration_fraction=EXPLORATION_FRACTION,
        exploration_final_eps=FINAL_EXPLORATION_DECAY,
        target_update_interval=1000,
        train_freq=WEIGHT_COLLECT_FREQ,
        gradient_steps=GRADIENT_UPDATE_FREQ,
        learning_starts=START_LEARNING_AT_STEP,
        batch_size=BATCH_SIZE,
    )

    # 4. Train
    print(f"Training for {ITERATIONS} steps...")
    model.learn(total_timesteps=ITERATIONS, tb_log_name=MODEL_NAME)
    
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

    for ep in range(10):
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
