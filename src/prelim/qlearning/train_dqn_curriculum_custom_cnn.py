import gymnasium as gym
import minigrid
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import DQN
import os
import sys
import time
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# --- Custom CNN for 7x7 MiniGrid inputs ---
class MinigridCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample())[None].float()).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))
# ------------------------------------------

# Directories - relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models", "dqn_curriculum")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Hyperparams
COMMON_PARAMS = dict(
    policy="CnnPolicy",
    verbose=1,
    tensorboard_log=LOG_DIR,
    device="cuda",
    buffer_size=100000,
    learning_rate=1e-4,
    policy_kwargs=dict(
        features_extractor_class=MinigridCNN,
        features_extractor_kwargs=dict(features_dim=128),
    ),
)

def train():
    print("--- STARTING DQN CURRICULUM TRAINING ---")
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # --- STAGE 1: Empty-8x8 (100k steps) ---
    print("\n>>> STAGE 1: Empty-8x8 (100k steps)")
    env = gym.make("MiniGrid-Empty-8x8-v0", render_mode="rgb_array")
    env = ImgObsWrapper(env)

    model = DQN(
        env=env,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        **COMMON_PARAMS
    )
    model.learn(total_timesteps=100000, tb_log_name="DQN_CurrS1_Empty8x8")
    model.save(os.path.join(MODELS_DIR, "S1_weights"))
    env.close()

    # --- STAGE 2: DoorKey-5x5 (100k steps) ---
    print("\n>>> STAGE 2: DoorKey-5x5 (100k steps)")
    env = gym.make("MiniGrid-DoorKey-5x5-v0", render_mode="rgb_array")
    env = ImgObsWrapper(env)

    model = DQN.load(os.path.join(MODELS_DIR, "S1_weights"), env=env, **COMMON_PARAMS)
    model.exploration_initial_eps = 0.2
    model.exploration_final_eps = 0.05
    model.exploration_fraction = 0.2
    
    model.learn(total_timesteps=100000, tb_log_name="DQN_CurrS2_DoorKey5x5")
    model.save(os.path.join(MODELS_DIR, "S2_weights"))
    env.close()

    # --- STAGE 3: DoorKey-8x8 (200k steps) ---
    print("\n>>> STAGE 3: DoorKey-8x8 (200k steps)")
    env = gym.make("MiniGrid-DoorKey-8x8-v0", render_mode="rgb_array")
    env = ImgObsWrapper(env)

    model = DQN.load(os.path.join(MODELS_DIR, "S2_weights"), env=env, **COMMON_PARAMS)
    model.exploration_initial_eps = 0.1
    model.exploration_final_eps = 0.05
    model.exploration_fraction = 0.1

    model.learn(total_timesteps=200000, tb_log_name="DQN_CurrS3Final_DoorKey8x8")
    model.save(os.path.join(MODELS_DIR, "S3_weights"))
    env.close()

    print("DQN CURRICULUM COMPLETE.")

def watch(stage, max_steps_per_episode=None):
    print(f"--- WATCHING STAGE {stage} ---")
    
    if stage == "1":
        model_path = os.path.join(MODELS_DIR, "S1_weights")
        env_id = "MiniGrid-Empty-8x8-v0"
    elif stage == "2":
        model_path = os.path.join(MODELS_DIR, "S2_weights")
        env_id = "MiniGrid-DoorKey-5x5-v0"
    elif stage == "3":
        model_path = os.path.join(MODELS_DIR, "S3_weights")
        env_id = "MiniGrid-DoorKey-8x8-v0"
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
    env = ImgObsWrapper(env)

    model = DQN.load(model_path, env=env)

    for ep in range(5):
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
                truncated = True # Force truncation if step limit reached
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
            print("Usage: python train_dqn_cur.py --watch <stage_number 1|2|3> [max_steps_per_episode]")
    else:
        train()