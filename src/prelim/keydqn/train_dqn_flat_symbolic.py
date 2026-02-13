import gymnasium as gym
import minigrid
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import DQN
import os
import sys
import time

# HYPER PARAMS
EXPLORATION_FRACTION = 0.5
FINAL_EXPLORATION_DECAY = 0.01
ITERATIONS = 200_000  # Bumped slightly to ensure convergence on the sparse reward
LEARNING_RATE = 0.0001
WATCH_EPISODES = 10
BUFFER = 15_000      # Lowered since you only train for 150k steps
WEIGHT_COLLECT_FREQ = 4
GRADIENT_UPDATE_FREQ = 4
START_LEARNING_AT_STEP = 1000
BATCH_SIZE = 128
TARGETNET_UPDATE_INTERVAL = 10000

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR))), "src", "models", "keydqn_flat")
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR))), "src", "logs", "keydqn_flat")
ENV_ID = "MiniGrid-DoorKey-5x5-v0"
MODEL_NAME = "DQN_Flat_DoorKey5x5"

def train():
    print(f"--- STARTING DQN FLAT/SYMBOLIC TRAINING ({ENV_ID}) ---")
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    def make_env():
        env = gym.make(ENV_ID, render_mode="rgb_array")
        env = FlatObsWrapper(env)
        env = Monitor(env)
        return env

    # PRO TIP: Run 4 environments in parallel to speed up data collection!
    env = DummyVecEnv([make_env for _ in range(4)])

    print(f"Observation Space: {env.observation_space.shape}")

    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        device="cuda",
        buffer_size=BUFFER,
        learning_rate=LEARNING_RATE,
        exploration_fraction=EXPLORATION_FRACTION,
        exploration_final_eps=FINAL_EXPLORATION_DECAY,
        target_update_interval=TARGETNET_UPDATE_INTERVAL,
        train_freq=WEIGHT_COLLECT_FREQ,
        gradient_steps=GRADIENT_UPDATE_FREQ,
        learning_starts=START_LEARNING_AT_STEP,
        batch_size=BATCH_SIZE,
    )

    print(f"Training for {ITERATIONS} steps...")
    model.learn(total_timesteps=ITERATIONS, tb_log_name=MODEL_NAME)
    
    save_path = os.path.join(MODELS_DIR, MODEL_NAME)
    model.save(save_path)
    print(f"Model saved to {save_path}")
    env.close()

def watch(max_steps_per_episode=None):
    print(f"--- WATCHING DQN FLAT AGENT ({ENV_ID}) ---")
    
    model_path = os.path.join(MODELS_DIR, MODEL_NAME)
    if not os.path.exists(model_path + ".zip"):
        print(f"Model not found at {model_path}.zip")
        return

    # Use a standard, non-vectorized env for watching to avoid auto-reset bugs
    env = gym.make(ENV_ID, render_mode="human")
    env = FlatObsWrapper(env)

    model = DQN.load(model_path, env=env)

    for ep in range(10):
        # Standard Gym v26+ reset yields (obs, info)
        obs, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        step_count = 0
        
        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            # Standard Gym v26+ step yields 5 values
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