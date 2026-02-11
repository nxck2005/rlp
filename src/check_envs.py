import gymnasium as gym
import minigrid

print("Checking available DoorKey environments...")
envs = [id for id in gym.envs.registry.keys() if "MiniGrid-DoorKey" in id]
for e in sorted(envs):
    print(e)

print("\nChecking available Empty environments...")
envs = [id for id in gym.envs.registry.keys() if "MiniGrid-Empty" in id]
for e in sorted(envs):
    print(e)
