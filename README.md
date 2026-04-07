# rlp: Reinforcement Learning Laboratory (CSE4037)

A comprehensive study of **Curriculum Learning** versus **Baseline Training** within the `MiniGrid` ecosystem. This project implements and visualises various RL architectures, ranging from standard DQNs to advanced parallelised Recurrent PPO agents.

## Project Taxonomy

The laboratory is divided into five core research categories:

1.  **DQN Architectures:** Pixel-based Deep Q-Networks exploring frame stacking and custom CNN kernels for sparse observations.
2.  **PPO Visual:** Pixel-based Proximal Policy Optimization focused on on-policy stability in 8x8 environments.
3.  **PPO Symbolic:** High-speed logical extraction using `FlatObsWrapper` for rapid convergence.
4.  **Recurrent PPO (RPPO):** LSTM-driven agents designed to solve POMDP (Partially Observable) challenges by maintaining temporal memory.
5.  **REPP2 Advanced:** Our peak architecture—parallelised Recurrent PPO using `SubprocVecEnv` for multi-stage sequence learning across complex room boundaries.

## Key Research Findings

*   **Memory as a Necessity:** Standard MLP policies suffer from "goldfish memory." LSTMs are critical for 8x8 maps to remember the location of keys/doors when they exit the agent's 7x7 field of view.
*   **Symbolic vs. Pixel:** Symbolic observations (object IDs) converge orders of magnitude faster than raw RGB pixels by bypassing the visual perception bottleneck.
*   **Temporal Context:** Frame stacking (4-frame buffer) significantly stabilises DQN performance on vision-based tasks.
*   **Curriculum Efficiency:** Staged training (e.g., Empty -> DoorKey 5x5 -> DoorKey 8x8) accelerates the discovery of sparse rewards in high-dimensional state spaces.

## Installation & Setup

This project uses `uv` for Python and `npm` for the React dashboard.

### 1. Clone & Python Environment
```bash
git clone https://github.com/nxck2005/rlp
cd rlp/src
uv sync
```

### 2. Frontend Environment
```bash
cd rlp/frontend
npm install
```

## Running the Laboratory (Webapp)

To launch the integrated visualizer, you must run both the backend and frontend in parallel sessions:

**Backend (from `rlp/src`):**
```bash
uv run uvicorn api.server:app --host 0.0.0.0 --port 8000
```

**Frontend (from `rlp/frontend`):**
```bash
npm run dev
```

*Dashboard available at: http://localhost:5173*

## Features

### Unified Visualizer Dashboard
A modern "Laboratory" interface built with React 19 + TypeScript to monitor agent logic in real-time.
*   **Dual Viewports:** Simultaneously view the Global God-View and the Agent's 7x7 Partial Observation.
*   **Spatial Analysis (Phase 3):**
    *   **Dynamic Heatmaps:** Visualise agent occupancy and exploration density layers.
    *   **Breadcrumb Tracking:** Real-time path tracking to identify navigation loops and efficiency.
*   **Episode Analysis:** Interactive "Time-Travel" scrubber to step through historical episodes frame-by-frame.
*   **Policy Telemetry:** Live action distribution charts and rolling success rate metrics.

---
Detailed theoretical notes and parameter explanations can be found in `CITED.md` and `PARAMS.md`.
