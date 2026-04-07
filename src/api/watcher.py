import gymnasium as gym
import numpy as np
import base64
from PIL import Image
import io
import os
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO, DQN
from minigrid.wrappers import FlatObsWrapper, ImgObsWrapper

class HeadlessWatcher:
    def __init__(self, env_id, model_path, model_type="RPPO", obs_type="Flat"):
        self.env_id = env_id
        self.model_path = model_path
        self.model_type = model_type
        self.obs_type = obs_type
        
        # Initialize environment in rgb_array mode
        self.env = gym.make(self.env_id, render_mode="rgb_array")
        
        if obs_type == "Flat":
            self.env = FlatObsWrapper(self.env)
        elif obs_type == "Pixel":
            self.env = ImgObsWrapper(self.env)
        
        # Load the appropriate model
        if model_type == "RPPO":
            self.model = RecurrentPPO.load(model_path)
        elif model_type == "PPO":
            self.model = PPO.load(model_path)
        elif model_type == "DQN":
            self.model = DQN.load(model_path)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        self.obs, _ = self.env.reset()
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
            
        # Take step
        self.obs, reward, terminated, truncated, info = self.env.step(action)
        self.total_reward += reward
        self.step_count += 1
        
        # Prepare for next step
        done = terminated or truncated
        self.episode_start = np.atleast_1d(done)
        
        # Capture frames
        global_frame = self.env.render()
        
        # Get the 7x7 observation image
        # We need the original image observation, not the flattened one
        try:
            # Unwrap to reach the Minigrid base env to call gen_obs()
            agent_view = self.env.unwrapped.gen_obs()['image'] # (7, 7, 3)
            full_grid = self.env.unwrapped.grid.encode() # (W, H, 3)
            agent_pos = self.env.unwrapped.agent_pos
            agent_dir = self.env.unwrapped.agent_dir
        except:
            # Fallback if gen_obs fails
            agent_view = np.zeros((7, 7, 3), dtype=int)
            full_grid = np.zeros((self.env.unwrapped.width, self.env.unwrapped.height, 3), dtype=int)
            agent_pos = (0, 0)
            agent_dir = 0
        
        if done:
            self.obs, _ = self.env.reset()
            self.lstm_states = None
            final_reward = self.total_reward
            steps = self.step_count
            self.total_reward = 0
            self.step_count = 0
            return global_frame, agent_view, full_grid, agent_pos, agent_dir, action, reward, done, True, final_reward, steps
            
        return global_frame, agent_view, full_grid, agent_pos, agent_dir, action, reward, done, False, 0, 0

    def get_frame_base64(self, frame):
        img = Image.fromarray(frame)
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def close(self):
        self.env.close()
