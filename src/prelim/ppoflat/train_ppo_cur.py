import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
import os

# --- HYPERPARAMETERS ---
LEARNING_RATE = 0.0003
ENTROPY_COEF = 0.05
BATCH_SIZE = 256
N_STEPS = 2048

# We create a helper function so we can easily swap environments
def make_env(env_id):
    def _init():
        env = gym.make(env_id, render_mode="rgb_array")
        env = FlatObsWrapper(env)
        env = Monitor(env)
        return env
    return _init

if __name__ == "__main__":
    print("--- PHASE 1: ELEMENTARY SCHOOL (Learn to Walk) ---")
    # 1. Start with the easiest possible environment
    env_easy = DummyVecEnv([make_env("MiniGrid-Empty-5x5-v0") for _ in range(4)])

    model = PPO(
        "MlpPolicy",
        env_easy,
        verbose=1,
        tensorboard_log="./curriculum_logs/",
        device="cpu",
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        ent_coef=ENTROPY_COEF,
    )

    # 2. Train on the easy task. 50k is more than enough for Empty-5x5.
    model.learn(total_timesteps=50_000, tb_log_name="Phase1_Empty")
    model.save("Phase1_Model")
    print("Phase 1 Model saved!")

    print("\n--- PHASE 2: HIGH SCHOOL (Learn the Door/Key Logic) ---")
    # 3. Load the target environment
    env_hard = DummyVecEnv([make_env("MiniGrid-DoorKey-5x5-v0") for _ in range(4)])

    model.set_env(env_hard)

    # 5. Continue training. The agent keeps its brain but faces the new challenge.
    model.learn(total_timesteps=150_000, tb_log_name="Phase2_DoorKey", reset_num_timesteps=False)
    
    model.save("Phase2_Model")
    print("Curriculum complete and model saved!")