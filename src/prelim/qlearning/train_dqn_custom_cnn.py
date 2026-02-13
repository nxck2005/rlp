import gymnasium as gym
import minigrid
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import DQN
import os
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# --- Custom CNN for 7x7 MiniGrid inputs ---
# Generated via Gemini, needs a bit of audit.
# Provided CNN's don't work with maze's odd kernel size
class MinigridCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        n_input_channels = observation_space.shape[0]
        
        self.cnn = nn.Sequential(
            # Layer 1: 3x3 kernel, seeing 3x3 local patterns (like corners/walls)
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # Layer 2: 3x3 kernel again to aggregate features
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # Layer 3: Another one for depth
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            # (7,7,3) becomes (3,7,7) because of VecTransposeImage, then
            # need to sample the observation space, convert to float tensor, 
            # and add a batch dimension for the CNN to process.
            # Then perform a forward pass to get the flattened size.
            n_flatten = self.cnn(th.as_tensor(observation_space.sample())[None].float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))
# ------------------------------------------

# Create directories
models_dir = os.path.join(os.path.dirname(__file__), "models", "dqn_baseline")
log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(models_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

print("Starting DQN baseline training on GPU with Custom 7x7 CNN...")

# Setup environment
env = gym.make("MiniGrid-DoorKey-8x8-v0", render_mode="rgb_array")
env = ImgObsWrapper(env) 

# Initialize DQN with Custom Policy
model = DQN(
    "CnnPolicy", 
    env, 
    verbose=1, 
    tensorboard_log=log_dir, 
    device="cuda", 
    buffer_size=100000,
    learning_rate=1e-4,
    exploration_fraction=0.1,
    exploration_final_eps=0.05,
    policy_kwargs=dict(
        features_extractor_class=MinigridCNN,
        features_extractor_kwargs=dict(features_dim=128),
    ),
)

# Total training steps
total_timesteps = 300000
print(f"Training for {total_timesteps} timesteps...")

model.learn(total_timesteps=total_timesteps, tb_log_name="DQN_Baseline_8x8_CustomCNN")

model.save(f"{models_dir}/dqn_8x8_model")
env.close()

print("DQN baseline training complete.")