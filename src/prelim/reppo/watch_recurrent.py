import gymnasium as gym
from sb3_contrib import RecurrentPPO
from minigrid.wrappers import FlatObsWrapper
import time

# 1. Load the environment (must match the training environment)
env = gym.make("MiniGrid-DoorKey-8x8-v0", render_mode="human")
env = FlatObsWrapper(env)

# 2. Load the trained model
MODEL_PATH = "models/rppo_baseline_final.zip"
model = RecurrentPPO.load(MODEL_PATH)

print("Watching the Baseline Agent... Press Ctrl+C to stop.")

obs, _ = env.reset()
# LSTM state must be initialized for Recurrent PPO
lstm_states = None
episode_start = True

try:
    while True:
        # Recurrent PPO requires passing the lstm_states
        action, lstm_states = model.predict(
            obs, 
            state=lstm_states, 
            episode_start=episode_start, 
            deterministic=True
        )
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        time.sleep(0.1) # Slows it down so you can see the logic

        episode_start = terminated or truncated
        if episode_start:
            print(f"Episode Finished! Reward: {reward}")
            obs, _ = env.reset()
            lstm_states = None # Reset memory for new episode
except KeyboardInterrupt:
    print("Closed.")
    env.close()