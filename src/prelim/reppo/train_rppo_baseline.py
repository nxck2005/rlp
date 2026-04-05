import gymnasium as gym
import os
import torch
from sb3_contrib import RecurrentPPO
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# 1. Setup Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs/rppo_baseline/")
MODEL_PATH = os.path.join(BASE_DIR, "models/rppo_baseline_final")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# 2. Environment Factory
def make_env():
    # Training directly on the hard 8x8 version
    env = gym.make("MiniGrid-DoorKey-8x8-v0", render_mode="rgb_array")
    env = FlatObsWrapper(env) # Symbolic 1D array
    env = Monitor(env, LOG_DIR) # Required for ep_rew_mean logs
    return env

# Use DummyVecEnv for stable-baselines compatibility
env = DummyVecEnv([make_env])

# 3. Initialize RPPO with CUDA
model = RecurrentPPO(
    "MlpLstmPolicy", # LSTM provides memory for POMDP
    env,
    learning_rate=0.0003, #
    n_steps=2048, #
    batch_size=256, #
    ent_coef=0.05, # Higher exploration for sparse rewards
    verbose=1,
    tensorboard_log=LOG_DIR,
    device="cuda"
)

# 4. Train and Save
print("Starting Baseline Training on CUDA...")
model.learn(total_timesteps=300000)
model.save(MODEL_PATH)
print(f"Baseline model saved to {MODEL_PATH}")