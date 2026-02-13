import gymnasium as gym
import minigrid
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO  # <--- Swapped to PPO
import os
import sys
import time

# --- PPO HYPERPARAMETERS ---
ITERATIONS = 100_000      
LEARNING_RATE = 0.0003     
ENTROPY_COEF = 0.05        # The "curiosity" metric. Prevents catastrophic forgetting!
BATCH_SIZE = 256           # How many experiences to look at once during an update
N_STEPS = 2048             # How many steps to collect per env before pausing to learn
N_EPOCHS = 10              # How many passes to make over the collected data
WATCH_EPISODES = 10

# Directories (Updated to keep PPO models separate from DQN)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR))), "src", "models", "keyppo_flat")
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR))), "src", "logs", "keyppo_flat")
ENV_ID = "MiniGrid-DoorKey-5x5-v0"
MODEL_NAME = "PPO_Flat_DoorKey5x5"

def train():
    print(f"--- STARTING PPO FLAT/SYMBOLIC TRAINING ({ENV_ID}) ---")
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    def make_env():
        env = gym.make(ENV_ID, render_mode="rgb_array")
        env = FlatObsWrapper(env)
        env = Monitor(env)
        return env

    # Run 4 environments in parallel to speed up data collection
    env = DummyVecEnv([make_env for _ in range(4)])

    print(f"Observation Space: {env.observation_space.shape}")

    # Swapped to PPO
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        device="cpu",
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        ent_coef=ENTROPY_COEF,
        n_epochs=N_EPOCHS,
    )

    print(f"Training for {ITERATIONS} steps...")
    model.learn(total_timesteps=ITERATIONS, tb_log_name=MODEL_NAME)
    
    save_path = os.path.join(MODELS_DIR, MODEL_NAME)
    model.save(save_path)
    print(f"Model saved to {save_path}")
    env.close()

def watch(max_steps_per_episode=None):
    print(f"--- WATCHING PPO FLAT AGENT ({ENV_ID}) ---")
    
    model_path = os.path.join(MODELS_DIR, MODEL_NAME)
    if not os.path.exists(model_path + ".zip"):
        print(f"Model not found at {model_path}.zip")
        return

    # Standard, non-vectorized env for watching
    env = gym.make(ENV_ID, render_mode="human")
    env = FlatObsWrapper(env)

    model = PPO.load(model_path, env=env)

    for ep in range(WATCH_EPISODES):
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