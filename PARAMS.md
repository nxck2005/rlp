### Memory & Learning (The "Brain")

*   **`buffer_size=100000`**:
    *   **What it is:** The **Replay Buffer** capacity. It stores the last 100,000 steps (observations, actions, rewards) the agent took.
    *   **Why it matters:** DQN learns by randomly sampling from this history to break correlations between consecutive steps.
    *   **Tuning:**
        *   **Larger:** Better stability, but consumes more RAM.
        *   **Smaller:** Learns faster from recent events but might "forget" older, important experiences (catastrophic forgetting).
        *   *Verdict:* 100k is a solid default. If your agent forgets how to open doors later in training, try increasing this.

*   **`learning_rate=1e-4` (0.0001)**:
    *   **What it is:** Step size for the optimizer (Adam).
    *   **Why it matters:** Controls how drastically the neural network weights change after each update.
    *   **Tuning:**
        *   **Higher (e.g., 1e-3):** Learns faster but might be unstable or diverge (fail completely).
        *   **Lower (e.g., 1e-5):** Very stable but takes forever to learn.
        *   *Verdict:* `1e-4` is the "gold standard" for DQN.

### Exploration (The "Curiosity")

*   **`exploration_fraction=0.2`**:
    *   **What it is:** The percentage of the total training time over which the exploration rate decays.
    *   **Context:** You are training for 500,000 steps. `0.2` means for the first **100,000 steps** (20%), the agent will gradually go from 100% random actions down to 5% random actions.
    *   **Tuning:**
        *   **Increase (e.g., 0.5):** If the agent never finds the key/door because the map is too big. It forces it to wander randomly for longer.
        *   **Decrease (e.g., 0.1):** If the task is simple and the agent is wasting time acting randomly when it already knows what to do.

*   **`exploration_final_eps=0.05`**:
    *   **What it is:** The final "Epsilon" (randomness) value.
    *   **Context:** After the decay period (above), the agent will stick to this value forever. `0.05` means the agent takes a random action **5% of the time**, even when it thinks it's an expert.
    *   **Why:** Prevents the agent from getting stuck in a loop (e.g., banging its head against a wall forever).

### Training Frequency (The "Rhythm")

*   **`train_freq=4`**:
    *   **What it is:** How often to update the neural network weights.
    *   **Context:** The agent takes **4 steps** in the game (collecting data), and *then* triggers a training update.
    *   **Tuning:**
        *   **Lower (1):** Updates every single step. Computationally expensive (slow wall-clock time) but data-efficient.
        *   **Higher (e.g., 16):** Faster simulation, but might learn slower in terms of game steps.
        *   *Verdict:* `4` is standard for Atari/Minigrid. Matches the frame-skip logic often used in papers.

*   **`gradient_steps=1`**:
    *   **What it is:** How many times to run the optimizer per `train_freq`.
    *   **Context:** After every 4 game steps (from `train_freq`), the model grabs **1 batch** from the replay buffer and updates weights once.
    *   **Tuning:**
        *   **Higher:** Extracts more juice from every experience, but slows down training significantly. Can sometimes lead to overfitting on old data.

*   **`target_update_interval=1000`**:
    *   **What it is:** How often to update the **Target Network**.
    *   **Why:** DQN actually uses *two* networks: the one it's training (Policy) and a stable copy (Target) used to calculate loss. This prevents the "dog chasing its tail" instability.
    *   **Context:** Every 1,000 steps, the Target network becomes an exact copy of the Policy network.
    *   **Tuning:**
        *   **Lower:** Faster feedback, but risks instability/oscillation.
        *   **Higher:** More stable, but learning might feel "laggy."

### Logging

*   **`verbose=1`**:
    *   Prints standard info to the console (FPS, mean reward, loss) during training.
*   **`tensorboard_log=LOG_DIR`**:
    *   Saves detailed metrics (loss curves, reward plots) to the specified folder for visualization in TensorBoard.

---

## PPO Parameters

### Core Components

*   **`"MlpPolicy"`**:
    *   **What it is:** The network architecture. For `FlatObsWrapper` (symbolic/vector inputs), a **Multi-Layer Perceptron (MLP)** is used. This is a standard feed-forward neural network.
    *   **Tuning:** Typically used for non-image observations. If using pixel inputs, you would use `"CnnPolicy"`.

*   **`env`**:
    *   **What it is:** The training environment (your `DummyVecEnv` wrapper around MiniGrid).
    *   **Note:** Defines the input shape and action space.

*   **`device="cpu"`**:
    *   **What it is:** Hardware acceleration.
    *   **Impact:** Forces the model to train on your CPU. While PPO can benefit from GPUs, for smaller MiniGrid environments with `MlpPolicy`, CPU can be sufficient and avoid GPU memory overhead if not needed.

### Learning & Optimization

*   **`learning_rate=0.0003`**:
    *   **What it is:** Step size for the optimizer (Adam).
    *   **Why it matters:** Controls how drastically the neural network weights change after each update.
    *   **Tuning:**
        *   **Higher (e.g., 1e-3):** Learns faster but might be unstable or diverge.
        *   **Lower (e.g., 1e-5):** Very stable but takes longer to learn.
        *   *Verdict:* `3e-4` (0.0003) is a common default for PPO and often performs well.

*   **`n_steps=2048`**:
    *   **What it is:** The number of steps the agent takes in the environment *before* performing a policy update. This is the length of the rollout buffer.
    *   **Why it matters:** PPO collects experience in batches (rollouts). A larger `n_steps` means more diverse experience in each update, potentially more stable updates but also less frequent updates.
    *   **Tuning:**
        *   **Larger:** Better gradient estimates, often more stable. Can require more RAM.
        *   **Smaller:** More frequent updates, but potentially noisier gradients.
        *   *Verdict:* 2048 is a standard value.

*   **`batch_size=256`**:
    *   **What it is:** The size of the mini-batches used to optimize the policy and value networks *during each update*. PPO typically performs multiple passes over the collected `n_steps` data, breaking it into `batch_size` chunks.
    *   **Why it matters:** Affects the stability and efficiency of the gradient descent.
    *   **Tuning:**
        *   **Larger:** Smoother gradients, less variance.
        *   **Smaller:** Noisier gradients, can help escape local optima. Must be a divisor of `n_steps`.
        *   *Verdict:* 256 is a reasonable size for PPO.

*   **`ent_coef=0.05`**:
    *   **What it is:** The **entropy coefficient**. This hyperparameter controls the weight given to the entropy bonus in the loss function.
    *   **Why it matters:** Entropy measures the randomness of the policy. A higher entropy coefficient encourages the agent to explore more (i.e., make its actions more random) rather than settling on a single deterministic action too early.
    *   **Tuning:**
        *   **Higher:** More exploration, can prevent premature convergence to suboptimal policies.
        *   **Lower (or 0):** Less exploration, allows the policy to become more deterministic quickly.
        *   *Verdict:* 0.05 is a good starting point. If the agent gets stuck in local optima, try increasing it.

### Logging

*   **`verbose=1`**:
    *   Prints standard info to the console (FPS, mean reward, loss) during training.
*   **`tensorboard_log="./curriculum_logs/"`**:
    *   Saves detailed metrics (loss curves, reward plots) to the specified folder for visualization in TensorBoard.