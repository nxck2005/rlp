import gymnasium as gym
import minigrid
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
import os

models_dir = "models/baseline"
log_dir = "logs"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

print("Starting baseline training..")
env = gym.make("MiniGrid-DoorKey-5x5-v0", render_mode="rgb_array")
env = ImgObsWrapper(env)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, device="cpu")

# We give it the SAME total time (180k steps)
model.learn(total_timesteps=180000, tb_log_name="Baseline_NoCurriculum")

model.save(f"{models_dir}/baseline_model")
env.close()

print("Baseline training done.")