import gymnasium as gym
import os
import torch
from sb3_contrib import RecurrentPPO
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# 1. Setup Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs/rppo_curriculum/")
os.makedirs(LOG_DIR, exist_ok=True)

# 2. Define Curriculum Stages
STAGES = [
    {"id": "MiniGrid-Empty-8x8-v0", "steps": 60000, "name": "1_nav"},
    {"id": "MiniGrid-DoorKey-5x5-v0", "steps": 80000, "name": "2_interact"},
    {"id": "MiniGrid-DoorKey-8x8-v0", "steps": 160000, "name": "3_target"}
]

model = None

# 3. Training Loop across Stages
for stage in STAGES:
    print(f"\n>>> Starting Stage: {stage['name']} ({stage['id']})")
    
    def make_env():
        env = gym.make(stage["id"], render_mode="rgb_array")
        env = FlatObsWrapper(env)
        env = Monitor(env, LOG_DIR)
        return env

    env = DummyVecEnv([make_env])

    if model is None:
        # Initial Brain Creation
        model = RecurrentPPO(
            "MlpLstmPolicy", 
            env, 
            learning_rate=0.0003,
            ent_coef=0.05,
            verbose=1,
            tensorboard_log=LOG_DIR,
            device="cuda" 
        )
    else:
        # Weight Transfer: Keep the brain, change the environment
        model.set_env(env)

    # reset_num_timesteps=False keeps the TensorBoard line continuous
    model.learn(total_timesteps=stage["steps"], reset_num_timesteps=False)
    
    # Save intermediate progress
    SRC_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
    if stage["name"] == "3_target":
        save_path = os.path.join(SRC_DIR, f"models/rppo_cur/rppo_cur_{stage['name']}")
    else:
        save_path = os.path.join(SRC_DIR, f"models/rppo_cur_temp/rppo_cur_{stage['name']}")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)

print("\nCurriculum Training Complete on CUDA.")