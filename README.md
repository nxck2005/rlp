# rlp: Reinforcement Learning Project (CSE4037)

Implementation and study of **Curriculum Learning** versus **Baseline Training** in Reinforcement Learning using `MiniGrid`.

## Project Overview

The objective is to evaluate the efficacy of staged training on progressively complex tasks. Our core experiments focus on the `MiniGrid-DoorKey-5x5-v0` environment, where an agent must navigate a grid, find a key, and unlock a door.

### Key Research Findings
*   **POMDP Nature:** `MiniGrid` is partially observable. Standard reactive agents (like MLP policies without memory) often struggle with sequences like "get key, then find door."
*   **FrameStacking Success:** Adding FrameStack (short-term temporal memory) to DQN agents significantly improved performance on `DoorKey-5x5-v0`.
*   **DQN vs. PPO:** 
    *   **DQN (Pixels):** Faced instability and catastrophic forgetting due to off-policy buffer poisoning in sparse reward settings.
    *   **PPO (Symbolic):** On-policy architecture with gradient clipping proved much more stable and converged faster, especially when using symbolic (`FlatObsWrapper`) observations.
*   **Policy Entrapment:** Some curriculum designs (Stage 1 Empty -> Stage 2 DistShift -> Stage 3 DoorKey) led to policy entrapment in the final stage, while baseline training eventually converged.

## Experimental Design

### 1. PPO Experiments (Symbolic Observations)
Located in `src/prelim/ppo/` and `src/prelim/ppoflat/`.
*   **Observation:** 1D symbolic array (`FlatObsWrapper`).
*   **Policy:** `MlpPolicy`.
*   **Focus:** Stable convergence and weight transfer between stages.

### 2. DQN Experiments (Pixel-based)
Located in `src/prelim/keydqn/` and `src/prelim/qlearning/`.
*   **Observation:** 56x56 RGB image (`RGBImgPartialObsWrapper`).
*   **Policy:** `CnnPolicy`.
*   **Specialized:** Includes experiments with **FrameStacking** to provide temporal context.

## Installation and Setup

This project uses `uv` for dependency management.

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd rlp
    ```
2.  **Install dependencies:**
    ```bash
    cd src
    uv sync
    ```

## Executing Experiments

All commands should be run from the `src/` directory.

### PPO (Proximal Policy Optimization)
*   **Baseline:** `uv run python prelim/ppo/train_ppo_baseline.py`
*   **Curriculum:** `uv run python prelim/ppo/train_ppo_curriculum.py`
*   **Watcher:** `uv run python prelim/ppo/watch_ppo.py`

### DQN (Deep Q-Learning)
*   **Pixel Baseline:** `uv run python prelim/keydqn/train_dqn_pixels_baseline.py`
*   **Pixel Curriculum:** `uv run python prelim/keydqn/train_dqn_pixels_curriculum.py`
*   **FrameStack Baseline:** `uv run python prelim/keydqn/train_dqn_pixels_framestack.py`
*   **Watch Pixel Agent:** `uv run python prelim/qlearning/watch_dqn_custom_cnn.py`

## Monitoring and Visualization

### TensorBoard
Monitor training progress (reward curves, loss, etc.):
```bash
cd src
uv run tensorboard --logdir prelim/logs/
```

### Watching Trained Agents
Many training scripts include a `--watch` flag or have a dedicated `watch_*.py` script.
*   Example for PPO: `uv run python prelim/ppo/watch_ppo.py`
*   Example for PPO Flat Curriculum: `uv run python prelim/ppoflat/watch_curriculum.py <stage_number>`

---
Detailed theoretical notes, parameter explanations, and source citations can be found in `CITED.md` and `PARAMS.md`.
