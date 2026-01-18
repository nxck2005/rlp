import gymnasium as gym
import minigrid
from minigrid.wrappers import ImgObsWrapper # skip text, keep pixels
from stable_baselines3 import PPO
import os

# Total steps/epochs = 180k

# save models
models_dir = "models/curriculum"
log_dir = "logs"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# easy stage
print("STARTING EASY TASKS")
env = gym.make("MiniGrid-Empty-6x6-v0", render_mode="rgb_array")
env = ImgObsWrapper(env)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, device="cpu")
model.learn(total_timesteps=30000, tb_log_name="CurrS1")
model.save(f"{models_dir}/S1_weights")
env.close()

# medium stage : add obstacles but load weights from easy-trained
print("STARTING MEDIUM TASKS")
env = gym.make("MiniGrid-DistShift1-v0", render_mode="rgb_array")
env = ImgObsWrapper(env)

# load learnt weights from S1
model = PPO.load(f"{models_dir}/S1_weights", env=env)
model.learn(total_timesteps=50000, tb_log_name="CurrS2")
model.save(f"{models_dir}/S2_weights")
env.close()

# hard stage
print("STARTING HARD TASKS")
env = gym.make("MiniGrid-DoorKey-5x5-v0", render_mode="rgb_array")
env = ImgObsWrapper(env)

model = PPO.load(f"{models_dir}/S2_weights", env=env)

# train the longest here
model.learn(total_timesteps=100000, tb_log_name="CurrS3Final")
model.save(f"{models_dir}/S3_weights")
env.close()

print("CURRICULUM TRAINING COMPLETE, exiting!")