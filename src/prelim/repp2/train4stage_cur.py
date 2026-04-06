from turtle import delay

import gymnasium as gym
import os
import torch
from sb3_contrib import RecurrentPPO
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs/rppo4Stage/")
os.makedirs(LOG_DIR, exist_ok=True)

STAGES = [
    {"id": "MiniGrid-Empty-8x8-v0",      "steps": 50000,   "name": "1_nav"},
    {"id": "MiniGrid-DoorKey-5x5-v0",    "steps": 200000,  "name": "2_interact"},
    {"id": "MiniGrid-DoorKey-8x8-v0",    "steps": 500000,  "name": "3_target"},
    {"id": "MiniGrid-MultiRoom-N2-S4-v0", "steps": 1000000, "name": "4_sequence"}
]

def make_env(env_id, rank):
    def _init():
        env = gym.make(env_id, render_mode="rgb_array")
        env = FlatObsWrapper(env)
        env = Monitor(env)
        return env
    return _init


if __name__ == "__main__":
    model = None
    NUM_CPU = 4 

    for stage in STAGES:
        print(f"\n>>> Starting Stage: {stage['name']} | Goal: {stage['steps']} steps")
        
        
        envs = SubprocVecEnv([make_env(stage["id"], i) for i in range(NUM_CPU)])

        if model is None:
            model = RecurrentPPO(
                
                "MlpLstmPolicy", 
                envs, 
                learning_rate=0.0003,
                n_steps=512,        
                batch_size=128,     
                ent_coef=0.05,      
                gae_lambda=0.95,    
                verbose=1,
                tensorboard_log=LOG_DIR,
                device="cuda"
            )
        else:
            model.set_env(envs)

        model.learn(total_timesteps=stage["steps"], reset_num_timesteps=False)
        
        save_path = os.path.join(BASE_DIR, f"models/fast_{stage['name']}")
        model.save(save_path)
        
        
        envs.close()

    print("\n4 Stage Training Complete.")