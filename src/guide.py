# Gymnasium test

import gymnasium as gym

# init the environment
# gymnasium is a API for markov decision processes

env = gym.make("LunarLander-v3", render_mode="human")

# reset env to generate first observation
observation, info = env.reset(seed=67)

for epoch in range(10000):
    # insert policy here, usually
    action = env.action_space.sample()

    # do a transition thru the env with the action
    # recieving next obs, reward, and if episode has terminated
    observation, reward, terminated, truncated, info = env.step(action)

    # if so, reset to start a new episode
    if terminated or truncated:
        observation, info = env.reset()

env.close()