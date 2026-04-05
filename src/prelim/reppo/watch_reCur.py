import gymnasium as gym
import os
import numpy as np
from sb3_contrib import RecurrentPPO
from minigrid.wrappers import FlatObsWrapper
import time
# 1. Configuration
# Change this to the specific stage model you want to watch
MODEL_NAME = "rppo_cur_3_target" 
ENV_ID = "MiniGrid-DoorKey-8x8-v0"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, f"models/{MODEL_NAME}")

def watch_agent():
    # 2. Recreate the Environment (must match training wrappers)
    # Using render_mode="human" allows you to see the window popup
    env = gym.make(ENV_ID, render_mode="human")
    env = FlatObsWrapper(env)

    # 3. Load the Trained Model
    if not os.path.exists(MODEL_PATH + ".zip"):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    model = RecurrentPPO.load(MODEL_PATH, env=env)
    print(f"Loaded model: {MODEL_NAME}")

    # 4. Evaluation Loop
    obs, _ = env.reset()
    
    # RecurrentPPO specific: Initialize LSTM states as None
    # The model will initialize them to zeros automatically on the first call
    lstm_states = None
    episode_starts = np.atleast_1d(True)

    terminated = False
    truncated = False
    try:
        obs, _ = env.reset()
        lstm_states = None
        episode_starts = np.atleast_1d(True)
        step_count = 0

        while True:
            # 1. Try setting deterministic=False if it's stuck in a loop
            action, lstm_states = model.predict(
                obs, 
                state=lstm_states, 
                episode_start=episode_starts, 
                deterministic=False 
            )

            obs, reward, terminated, truncated, info = env.step(action)
            episode_starts = np.atleast_1d(terminated or truncated)
            
            step_count += 1
            if step_count % 10 == 0:
                print(f"Step: {step_count} | Action: {action}")

            # 2. Add a tiny sleep so the UI can refresh
            time.sleep(0.05) 

            if terminated or truncated:
                print(f"--- Episode Finished in {step_count} steps ---")
                obs, _ = env.reset()
                lstm_states = None
                episode_starts = np.atleast_1d(True)
                step_count = 0  

    except KeyboardInterrupt:
        print("\nClosing...")
    finally:
        env.close()

if __name__ == "__main__":
    watch_agent()