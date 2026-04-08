import gymnasium as gym
import torch as th
import torch.nn as nn
import numpy as np
import base64
from PIL import Image
import io
import os
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from minigrid.wrappers import FlatObsWrapper, ImgObsWrapper, RGBImgPartialObsWrapper

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

class HeadlessWatcher:
    def __init__(self, env_id, model_path, model_type="RPPO", obs_type="Flat", use_pixel_wrapper=False, n_stack=None):
        self.env_id = env_id
        self.model_path = model_path
        self.model_type = model_type
        self.obs_type = obs_type
        self.use_pixel_wrapper = use_pixel_wrapper
        self.n_stack = n_stack
        
        # Helper to create env
        def make_env():
            env = gym.make(self.env_id, render_mode="rgb_array")
            if self.use_pixel_wrapper:
                env = RGBImgPartialObsWrapper(env, tile_size=8)
            
            if self.obs_type == "Flat":
                env = FlatObsWrapper(env)
            elif self.obs_type == "Pixel":
                env = ImgObsWrapper(env)
            return env

        # Always use DummyVecEnv to unify the API (SB3 works best this way)
        self.env = DummyVecEnv([make_env])
        
        if self.obs_type == "Pixel":
            # SB3 CNN policies require (C, H, W)
            self.env = VecTransposeImage(self.env)
        
        if n_stack:
            self.env = VecFrameStack(self.env, n_stack=n_stack)
        
        # Custom objects for SB3 load
        custom_objects = {
            "features_extractor_class": MinigridCNN
        }

        # Load the appropriate model
        if model_type == "RPPO":
            self.model = RecurrentPPO.load(model_path, custom_objects=custom_objects)
        elif model_type == "PPO":
            self.model = PPO.load(model_path, custom_objects=custom_objects)
        elif model_type == "DQN":
            self.model = DQN.load(model_path, custom_objects=custom_objects)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        # VecEnv.reset() returns just the observation
        self.obs = self.env.reset()
        self.lstm_states = None
        self.episode_start = np.atleast_1d(True)
        self.step_count = 0
        self.total_reward = 0

    def step(self):
        # Handle predictions based on model type
        if self.model_type == "RPPO":
            action, self.lstm_states = self.model.predict(
                self.obs,
                state=self.lstm_states,
                episode_start=self.episode_start,
                deterministic=False
            )
        else:
            action, _ = self.model.predict(self.obs, deterministic=False)
            
        # VecEnv.step() returns (obs, rewards, dones, infos)
        self.obs, reward, done, info = self.env.step(action)
        
        # Extract scalar values from arrays (VecEnv always returns arrays of size 1)
        act_val = action[0]
        rew_val = reward[0]
        done_val = done[0]

        self.total_reward += rew_val
        self.step_count += 1
        
        # Prepare for next step
        self.episode_start = np.atleast_1d(done_val)
        
        # Capture frames
        base_env = self.env.envs[0].unwrapped
        global_frame = self.env.render()
        
        # Get the 7x7 observation image
        try:
            agent_view = base_env.gen_obs()['image'] # (7, 7, 3)
            full_grid = base_env.grid.encode() # (W, H, 3)
            agent_pos = base_env.agent_pos
            agent_dir = base_env.agent_dir
        except:
            # Fallback
            agent_view = np.zeros((7, 7, 3), dtype=int)
            full_grid = np.zeros((base_env.width, base_env.height, 3), dtype=int)
            agent_pos = (0, 0)
            agent_dir = 0
        
        if done_val:
            self.obs = self.env.reset()
            self.lstm_states = None
            final_reward = self.total_reward
            steps = self.step_count
            self.total_reward = 0
            self.step_count = 0
            return global_frame, agent_view, full_grid, agent_pos, agent_dir, act_val, rew_val, done_val, True, final_reward, steps
            
        return global_frame, agent_view, full_grid, agent_pos, agent_dir, act_val, rew_val, done_val, False, 0, 0

    def get_frame_base64(self, frame):
        img = Image.fromarray(frame)
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def close(self):
        self.env.close()
