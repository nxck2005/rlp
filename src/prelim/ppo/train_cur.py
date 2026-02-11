import gymnasium as gym
import minigrid
from minigrid.wrappers import ImgObsWrapper # skip text, keep pixels
from stable_baselines3 import PPO
import os

# Total steps/epochs = 300k

# save models
models_dir = "models/curriculum"
log_dir = "logs"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Stage 1: Learn to navigate the large room (Empty 8x8)
print("STARTING STAGE 1: Empty-8x8 (Navigation)")
env = gym.make("MiniGrid-Empty-8x8-v0", render_mode="rgb_array")
env = ImgObsWrapper(env)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, device="cpu")
model.learn(total_timesteps=50000, tb_log_name="CurrS1_Empty8x8")
model.save(f"{models_dir}/S1_weights")
env.close()

# Stage 2: Learn the mechanic in a small room (DoorKey 5x5)
print("STARTING STAGE 2: DoorKey-5x5 (Mechanic)")
env = gym.make("MiniGrid-DoorKey-5x5-v0", render_mode="rgb_array")
env = ImgObsWrapper(env)

# load learnt weights from S1
model = PPO.load(f"{models_dir}/S1_weights", env=env)
model.learn(total_timesteps=100000, tb_log_name="CurrS2_DoorKey5x5")
model.save(f"{models_dir}/S2_weights")
env.close()

# Stage 3: Target Task (DoorKey 8x8)
print("STARTING STAGE 3: DoorKey-8x8 (Target)")
env = gym.make("MiniGrid-DoorKey-8x8-v0", render_mode="rgb_array")
env = ImgObsWrapper(env)

model = PPO.load(f"{models_dir}/S2_weights", env=env)

# Train the rest of the budget here
model.learn(total_timesteps=150000, tb_log_name="CurrS3Final_DoorKey8x8")
model.save(f"{models_dir}/S3_weights")
env.close()

print("CURRICULUM TRAINING COMPLETE, exiting!")
