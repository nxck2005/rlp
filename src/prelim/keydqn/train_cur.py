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
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR))), "src", "models", "keydqn_cur")
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR))), "src", "logs", "keydqn_cur")

def train():
    print("--- STARTING DQN PIXEL CURRICULUM TRAINING ---")
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # --- STAGE 1: Empty-6x6 (100k steps) ---
    print("\n>>> STAGE 1: Empty-6x6 (100k steps)")
    env = gym.make("MiniGrid-Empty-6x6-v0", render_mode="rgb_array")
    env = RGBImgPartialObsWrapper(env, tile_size=8)
    env = ImgObsWrapper(env)

    model = DQN(
        "CnnPolicy",
        env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        device="cuda",
        buffer_size=100000,
        learning_rate=1e-4,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        target_update_interval=1000,
        train_freq=4,
        gradient_steps=1,
    )
    model.learn(total_timesteps=100000, tb_log_name="S1_Empty6x6")
    model.save(os.path.join(MODELS_DIR, "S1_weights"))
    env.close()

    # --- STAGE 2: DoorKey-5x5 (400k steps) ---
    print("\n>>> STAGE 2: DoorKey-5x5 (400k steps)")
    env = gym.make("MiniGrid-DoorKey-5x5-v0", render_mode="rgb_array")
    env = RGBImgPartialObsWrapper(env, tile_size=8)
    env = ImgObsWrapper(env)

    # Load S1 weights but attach to new env
    model = DQN.load(os.path.join(MODELS_DIR, "S1_weights"), env=env)
    
    # Reset exploration slightly for new task? Usually kept low or reset.
    # Let's keep it low (0.05) to exploit learned navigation, but maybe bump slightly to 0.1?
    model.exploration_initial_eps = 0.1
    model.exploration_final_eps = 0.05
    model.exploration_fraction = 0.1
    
    model.learn(total_timesteps=400000, tb_log_name="S2_DoorKey5x5")
    model.save(os.path.join(MODELS_DIR, "S2_final"))
    env.close()

    print("DQN CURRICULUM COMPLETE.")

def watch(stage, max_steps_per_episode=None):
    print(f"--- WATCHING STAGE {stage} ---")
    
    if stage == "1":
        model_path = os.path.join(MODELS_DIR, "S1_weights")
        env_id = "MiniGrid-Empty-6x6-v0"
    elif stage == "2":
        model_path = os.path.join(MODELS_DIR, "S2_final")
        env_id = "MiniGrid-DoorKey-5x5-v0"
    else:
        print(f"Unknown stage: {stage}")
        return

    if not os.path.exists(model_path + ".zip"):
        print(f"Model not found at {model_path}.zip")
        return

    try:
        env = gym.make(env_id, render_mode="human")
    except Exception as e:
        print(f"Error creating env: {e}")
        return
    
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
        if len(sys.argv) > 2:
            stage = sys.argv[2]
            max_steps = int(sys.argv[3]) if len(sys.argv) > 3 else None
            watch(stage, max_steps)
        else:
            print("Usage: python train_cur.py --watch <stage_number 1|2> [max_steps]")
    else:
        train()
