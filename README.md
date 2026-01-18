# rlp
Implementation for project component for CSE4037

## Project Overview

This repository presents a project and study of **Curriculum Learning** versus a traditional **Baseline Training** approach in the context of Reinforcement Learning. The objective is to evaluate the efficacy of staged training on progressively complex tasks, then try to adapt it into a general ML project.

Sources and the study we did is noted down in CITED.md for future report drafting.

## Experimental Design

The primary objective is to train an agent to solve the `MiniGrid-DoorKey-5x5-v0` environment, which is basically a grid with walls, a key and a door for an agent to open with a key.

*   **Agent Policy:** Proximal Policy Optimization (PPO).
*   **Total Training Timesteps:** Both methodologies are allocated a total of 180,000 timesteps to ensure a rigorous and equivalent comparison.

### 1. Baseline Training Approach (`train_dumb.py`)
This approach involves direct training of the agent on the `MiniGrid-DoorKey-5x5-v0` environment for the entire duration of 180,000 timesteps. This serves as the baseline for the experiment.

### 2. Curriculum Learning Approach (`train_cur.py`)
This methodology implements a three-stage curriculum, where the agent's learned weights are transferred sequentially between environments of increasing complexity:
1.  **Stage 1:** `MiniGrid-Empty-6x6-v0` (Basic navigation) - 60,000 timesteps
2.  **Stage 2:** `MiniGrid-DistShift1-v0` (Navigation with environmental shifts) - 60,000 timesteps
3.  **Stage 3:** `MiniGrid-DoorKey-5x5-v0` (Target task) - 60,000 timesteps

## Installation and Setup

### Recommended Environment
It is highly recommended to utilize a dedicated Windows Subsystem for Linux (WSL) environment. A WSL distribution running Ubuntu with CUDA drivers properly configured will provide the best development and execution experience for this project.

Right now, the trainers are set up to use CPU as MlpPolicy is being utilised, but for larger, general tasks in the future we'll switch to CNN, for which a proper CUDA setup will help with GPU utilization.

Detailed guidance on setting up WSL2 and with Ubuntu and CUDA can be found in the official NVIDIA documentation:
[NVIDIA CUDA on WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)


This project leverages `uv` for efficient dependency management. Alternatively, standard `pip` installation is supported.

1.  **Repository Cloning:**
    ```bash
    git clone <repository_url>
    cd rlp
    ```

2.  **Dependency Installation:**
    Navigate to the `src` directory and synchronize dependencies:
    ```bash
    cd src
    uv sync
    ```
    Alternatively, using `pip` for direct installation:
    ```bash
    pip install gymnasium minigrid stable-baselines3 shimmy tensorboard
    ```

3. If some other dependencies are missing (we haven't completed documentation!), add them via ```uv pip install x```.

### Executing Baseline Training
To initiate training under the baseline methodology:
```bash
python src/prelim/train_dumb.py
```

### Executing Curriculum Learning
To initiate training under the curriculum learning methodology:
```bash
python src/prelim/train_cur.py
```

## Experiment Outputs

Training progress and model checkpoints are systematically recorded:

*   **Logs (TensorBoard Compatible):** Stored in `src/prelim/logs/`.
    Visualization of training metrics can be achieved by running:
    ```bash
    uv run tensorboard --logdir src/prelim/logs/
    ```
*   **Model Checkpoints:** Saved within `src/prelim/models/`.
    *   `baseline/`: Contains models generated during baseline training.
    *   `curriculum/`: Contains models from each respective stage of curriculum training.