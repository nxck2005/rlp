# Preliminary research:

## Theory
What Is Curriculum Learning in RL? : https://milvus.io/ai-quick-reference/what-is-curriculum-learning-in-reinforcement-learning

## Documentation
Gymnasium Documentation : https://gymnasium.farama.org/index.html

Gymnasium defines the standard API for interacting with a MDP. It enforces a strict contract for agents acting in a world, regardless of the specifics of the said world.

- Mandates that every env has a step() and reset() method.
- uses ```spaces``` module to strictly type inputs and outputs.
- Can wrap environments to preprocess data.

This will be our base module for doing RL work. This will enable reproducible results and interop for different environments.