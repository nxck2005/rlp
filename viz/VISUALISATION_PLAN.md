# RL Visualization Dashboard: Technical Plan

This document outlines the architecture and implementation strategy for a unified web-based dashboard to monitor, visualize, and compare the reinforcement learning agents in this project.

## 1. Project Objective
Build a real-time, interactive web application that allows users to:
*   Select between four core models (DQN, PPO, RPPO Baseline, RPPO Curriculum).
*   Watch live agent gameplay via WebSocket streaming.
*   Analyze internal state (Agent-Eye View, Action Distribution, Telemetry).
*   Compare performance across different training methods side-by-side.

## 2. Technical Stack
*   **Frontend:** React (TypeScript) + Vite + TailwindCSS.
*   **Backend:** Python + FastAPI (handling WebSockets and model execution).
*   **Communication:** WebSockets for 30FPS live streaming of RGB arrays and telemetry.
*   **Data Visualization:** Chart.js or Recharts for live action distributions and success rates.

## 3. Core Features & "The 3 Views"

### View 1: The Global Perspective (The "What")
*   **Live Stream:** A high-resolution render of the MiniGrid environment.
*   **Heatmap Overlay:** Dynamic transparency grid showing agent occupancy.
*   **Path Tracking:** Visual line showing the agent's movement history in the current episode.

### View 2: The Agent's Perspective (The "Why")
*   **Agent-Eye View:** A 7x7 grid rendering of the `obs['image']` array.
*   **Saliency/Attention:** Highlighting which objects (Key, Door, Wall) are currently in the agent's field of view.
*   **LSTM State (for RPPO):** A simplified visualization of the hidden state "memory" activity.

### View 3: Performance Telemetry (The "Process")
*   **Action Distribution:** Real-time bar chart showing the frequency of actions (Turn, Move, Pick Up, Toggle).
*   **Success Metrics:** Rolling success rate (% of last 100 episodes solved).
*   **Reward Curve:** Live-updating plot of episodic rewards.

## 4. Implementation Phases

### Phase 1: Backend Infrastructure (FastAPI Bridge)
*   Implement a WebSocket server in `src/api/server.py`.
*   Create a "Headless Watcher" class that runs Gymnasium environments in `rgb_array` mode.
*   Build a model loader for `.zip` (SB3) and `.pt` (PyTorch) files.

### Phase 2: Frontend Scaffolding (React + Vite)
*   Initialize the Vite project in `/frontend`.
*   Build the Layout (Sidebar for model selection, Main Grid for views).
*   Implement the WebSocket hook to handle binary/JSON data streams.

### Phase 3: Analytical Data API
*   Endpoints to parse existing TensorBoard logs from `src/logs/`.
*   Logic to calculate occupancy heatmaps from historical coordinate data.

### Phase 4: Comparative Mode
*   Feature to launch two concurrent environments for side-by-side "racing" between Baseline and Curriculum models.

## 5. Execution Order
1.  **Environment Setup:** Initialize Vite and FastAPI.
2.  **Streaming Proof-of-Concept:** Get a single agent rendering on a web canvas.
3.  **UI/UX Polish:** Build the sidebars, charts, and telemetry panels.
4.  **Integration:** Add support for all 4 models and log parsing.
