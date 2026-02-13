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

Result from cur train: While the curriculum accelerated learning in intermediate tasks (Stage 2), it caused policy entrapment in the final stage, leading to failure, whereas the baseline successfully converged.


The agent does not see a simple coordinate list. It "sees" a 3D volume of data, like a low-resolution image.

*   **Structure:** A 3D grid of shape **(7, 7, 3)**.
    *   **7x7 Grid:** The agent sees a 7x7 square area in front of itself (including walls).
    *   **3 Channels:** Each cell in that 7x7 grid has 3 numbers describing it:
        1.  **Object ID:** (e.g., 0=Wall, 1=Floor, 2=Door, 5=Key, 8=Goal).
        2.  **Color:** (e.g., Red, Green, Blue).
        3.  **State:** (e.g., Open/Closed for doors, Locked/Unlocked).
*   **Extra Info:** It also gets a `direction` (0-3) telling it which way it's facing.

**Why not just (x,y)?** If the agent only knew "I am at (3,3)", it wouldn't know there is a wall in front of it or a key to its left. It needs to "see" the surroundings to learn "If Wall in front -> Turn".

#### The Policy Space (Action Space)

The Policy outputs one of these integers at every step:
*   `0`: Turn Left
*   `1`: Turn Right
*   `2`: Move Forward
*   `3`: Pick Up (object in front)
*   `4`: Drop (object being carried)
*   `5`: Toggle (open door/interact)
*   `6`: Done (optional signal)

**Size of Policy Space:** It is 7^epochs. If the agent takes 100 steps, there are 7^100 possible sequences of moves. This is astronomically large, which is why we need AI to find the best sequence.

#### 3. The Reward Function
This is the "Rules of the Game". In MiniGrid, the reward is **Sparse** (you get nothing until you win).

$$ R = 1 - 0.9 \times \frac{\text{step\_count}}{\text{max\_steps}} $$

*   **If you fail (hit max steps without reaching goal):** Reward = `0`.
*   **If you win:** You get a positive reward, but it **shrinks** the longer you take.
    *   *Fast win:* High reward (close to 1.0).
    *   *Slow win:* Low reward (close to 0.1).

**The Learning Dynamic:**
Because the reward is 0 for almost everything (walking, turning, hitting walls), the agent starts by flailing randomly. It will likely fail thousands of times. Eventually, by pure luck, it will stumble upon the goal. *FLASH!* It gets a reward of +0.8. The math (PPO algorithm) then "backpropagates" this success, making the actions that led to it slightly more likely to happen next time.

---

### Clarification: State vs. Observation

You're right to ask for exact details! The "state" or more accurately, the "observation" the agent receives in `MiniGrid` is a rich representation, not just a simple pair of coordinates.

Here's exactly what the agent "sees" and receives as its observation:

The observation is a **dictionary** with three components: `image`, `direction`, and `mission`. For the PPO agent, the most critical components it learns from are the `image` and `direction`.

1.  ### `image` Component
This is the core visual input, represented as a 3D NumPy array with shape `(7, 7, 3)`. Think of it as a small, localized "camera view" from the agent's perspective. Each of the 49 cells (7x7) in this view is described by 3 channels:

    *   **Channel 0: Object Type (IDs)**
        *   For each cell in the 7x7 view, this channel specifies *what kind of object* is in that cell. These are integer IDs.
        *   **Examples:**
            *   `0`: Empty space
            *   `1`: Wall
            *   `2`: Floor
            *   `3`: Door
            *   `4`: Key
            *   `5`: Ball
            *   `6`: Box
            *   `7`: Goal
            *   `8`: Lava
            *   `9`: Agent (itself, typically at the center of the view)
        *   This tells the agent about the physical layout and interactable elements.

    *   **Channel 1: Color (IDs)**
        *   For each cell, this channel specifies the *color* of the object identified in Channel 0. These are also integer IDs.
        *   **Examples:**
            *   `0`: Red
            *   `1`: Green
            *   `2`: Blue
            *   `3`: Purple
            *   `4`: Yellow
            *   `5`: Grey
        *   Color can be important for distinguishing keys from other keys, or specific doors.

    *   **Channel 2: State/Extra (IDs)**
        *   This channel provides *additional information* about the object's state or properties.
        *   **Examples:**
            *   For a `Door` object:
                *   `0`: Closed
                *   `1`: Open
                *   `2`: Locked (requiring a key)
            *   For a `Key` object:
                *   `0`: Unpicked
                *   `1`: Picked up (being carried by agent, though this would typically not be in the grid view)
        *   This allows the agent to know if a door is locked or already open, for instance.

2.  ### `direction` Component
This is a single integer value from `0` to `3`, indicating the agent's current orientation (which way it's facing) relative to the global grid.
    *   `0`: Facing right (East)
    *   `1`: Facing down (South)
    *   `2`: Facing left (West)
    *   `3`: Facing up (North)
    This is crucial for interpreting its `image` view and deciding which "forward" action means.

3.  ### `mission` Component
This is a string that provides a natural language description of the task (e.g., "find the red ball", or for your task, likely something like "open the yellow door with the key").
    *   **Important Note for your project:** While this is part of the observation dictionary, standard `stable-baselines3` PPO with its default `MlpPolicy` typically **does not use this string directly** as input to the neural network for learning. The policy usually only processes the flattened `image` array and the `direction` integer. Advanced policies *can* be built to parse and use mission strings, but that's not the default setup in this project.

So, when the agent makes a decision, it's not just getting `(3,2)`. It's getting a small "picture" of its immediate surroundings, detailed by object types, colors, and states, plus its current heading. This allows for complex reactive behaviors like "If I see a locked red door, and I have a red key, and I am facing the door, then Toggle."

---

### Theoretical Distinction: State vs. Observation

You caught me being imprecise. In strict Reinforcement Learning theory, they are **NOT** the same.

Here is the exact distinction:

### 1. The State (God's View)
The **State ($S_t$)** represents the **complete, objective truth** of the entire universe at time $t$. If you knew the State, you would know absolutely everything.

*   **In your project (MiniGrid):** The State is the **entire** grid layout.
    *   It includes the `(x,y)` coordinates of the agent.
    *   It includes the `(x,y)` coordinates of the Key and the Door.
    *   It includes the map of every wall in the 5x5 room.
    *   It includes what the agent is holding.

If the agent had access to the **State**, it would know exactly where the key is, even if the key is behind a wall.

### 2. The Observation (Agent's View)
The **Observation ($O_t$)** is a **partial filter** of the state. It is the only data the agent actually receives.

*   **In your project:** The agent is **Partially Observable**.
    *   It **cannot** see the whole grid.
    *   It only sees that 7x7 square in front of its nose (the "image" I described earlier).
    *   It does **not** know its global `(x,y)` coordinates. It only knows what is immediately visible.

### Why I confused them (and why the code does too)
In practical code (like `stable-baselines3`), the mathematical function for the Policy is defined as $\pi(a|s)$ (Policy given State).

However, because we rarely give the agent the full God-View State (that would be cheating!), we feed the **Observation** into the function instead.

So, in your Python scripts:
*   **Variable Name:** `observation` (Correct)
*   **Math Concept:** Treated as $S$ (State) for the sake of the PPO algorithm.

### Summary
*   **State:** The Global Map + `(x,y)` coords. (The Agent **never** sees this).
*   **Observation:** The 7x7 pixel grid + `direction`. (This is what I described in the previous turn).

The "richness" I mentioned refers to the **Observation**. Even though it is a partial view, it is complex (colors, object types) compared to a simple "I am at x=3, y=5".

---

### Theoretical Implications: Feeding Observation vs. State

That is an excellent question. It hits the fundamental boundary between "Standard RL" and "Real-world RL."

Theoretically, moving from **State** to **Observation** changes the problem from a **Markov Decision Process (MDP)** to a **Partially Observable Markov Decision Process (POMDP)**.

Here are the three major theoretical shifts that happen:

### 1. The Loss of the "Markov Property" (The "Amnesia" Problem)
The core assumption of most RL algorithms (including PPO) is the **Markov Property**:
> "The future depends only on the present, not on the past."

*   **With State ($S_t$):** If you know the grid map, knowing where you were 5 minutes ago is irrelevant. You have all the info you need right now to make the perfect move.
*   **With Observation ($O_t$):** The Markov property breaks.
    *   *Example:* Imagine your agent is facing a white wall.
        *   Scenario A: It is at the start of the maze.
        *   Scenario B: It is at the end of the maze, but facing the wrong way.
    *   The observation (white wall) is **identical** in both cases.
    *   The correct action depends on *how it got there* (History). But since we only feed it the current $O_t$, PPO treats Scenario A and B as the exact same situation and will force the agent to take the exact same action in both, even if one leads to a win and one leads to death.

### 2. State Aliasing (Perceptual Blindness)
This is the formal term for the example above. Multiple different true **States** map to the same **Observation**.
$$ S_1 \neq S_2 \quad \text{but} \quad O(S_1) = O(S_2) $$

*   **Theoretical Impact:** The optimal policy is no longer deterministic.
    *   If you are in State 1, the best move is "Left".
    *   If you are in State 2, the best move is "Right".
    *   Since the agent only sees "White Wall" (which could be either), it cannot learn "Always go Left" or "Always go Right."
    *   It is forced to learn a **Stochastic Policy** (e.g., "50% Left, 50% Right") to hedge its bets, or a "safe" suboptimal policy (e.g., "Turn around").

### 3. The Need for Memory (Reactive vs. Recurrent)
Because the current observation $O_t$ is not enough to define the situation, the *true* optimal policy theoretically requires the **History of Observations**.

$$ \pi^*(a | o_t, o_{t-1}, ... o_0) $$

*   **What your code does (`MlpPolicy`):** It uses a Feed-Forward Network. It has **no memory**. It is a **Reactive Agent**. It decides solely on what it sees *right now*.
    *   *Result:* It acts like a goldfish. If it turns 360 degrees, it forgets what it saw at the start of the turn.
*   **What is theoretically required (`LstmPolicy`):** To solve a POMDP perfectly, you theoretically need a Recurrent Neural Network (LSTM or GRU). This gives the agent an internal "hidden state" (memory) that acts as a proxy for the missing True State.

### Summary for your Project
By feeding **Observation** to a memory-less PPO (`MlpPolicy`):
1.  You are theoretically solving a **POMDP** with an improper tool (a reactive policy).
2.  The agent will likely develop "Wall Hugging" or "Random Walk" behaviors. It cannot "map" the room in its head. It has to physically keep the goal in sight or follow walls to navigate, because if it looks away, the goal ceases to exist for the agent.

### Theoretical Distinction: State vs. Observation

You're right to ask for exact details! The "state" or more accurately, the "observation" the agent receives in `MiniGrid` is a rich representation, not just a simple pair of coordinates.

Here's exactly what the agent "sees" and receives as its observation:

The observation is a **dictionary** with three components: `image`, `direction`, and `mission`. For the PPO agent, the most critical components it learns from are the `image` and `direction`.

1.  ### `image` Component
This is the core visual input, represented as a 3D NumPy array with shape `(7, 7, 3)`. Think of it as a small, localized "camera view" from the agent's perspective. Each of the 49 cells (7x7) in this view is described by 3 channels:

    *   **Channel 0: Object Type (IDs)**
        *   For each cell in the 7x7 view, this channel specifies *what kind of object* is in that cell. These are integer IDs.
        *   **Examples:**
            *   `0`: Empty space
            *   `1`: Wall
            *   `2`: Floor
            *   `3`: Door
            *   `4`: Key
            *   `5`: Ball
            *   `6`: Box
            *   `7`: Goal
            *   `8`: Lava
            *   `9`: Agent (itself, typically at the center of the view)
        *   This tells the agent about the physical layout and interactable elements.

    *   **Channel 1: Color (IDs)**
        *   For each cell, this channel specifies the *color* of the object identified in Channel 0. These are also integer IDs.
        *   **Examples:**
            *   `0`: Red
            *   `1`: Green
            *   `2`: Blue
            *   `3`: Purple
            *   `4`: Yellow
            *   `5`: Grey
        *   Color can be important for distinguishing keys from other keys, or specific doors.

    *   **Channel 2: State/Extra (IDs)**
        *   This channel provides *additional information* about the object's state or properties.
        *   **Examples:**
            *   For a `Door` object:
                *   `0`: Closed
                *   `1`: Open
                *   `2`: Locked (requiring a key)
            *   For a `Key` object:
                *   `0`: Unpicked
                *   `1`: Picked up (being carried by agent, though this would typically not be in the grid view)
        *   This allows the agent to know if a door is locked or already open, for instance.

2.  ### `direction` Component
This is a single integer value from `0` to `3`, indicating the agent's current orientation (which way it's facing) relative to the global grid.
    *   `0`: Facing right (East)
    *   `1`: Facing down (South)
    *   `2`: Facing left (West)
    *   `3`: Facing up (North)
    This is crucial for interpreting its `image` view and deciding which "forward" action means.

3.  ### `mission` Component
This is a string that provides a natural language description of the task (e.g., "find the red ball", or for your task, likely something like "open the yellow door with the key").
    *   **Important Note for your project:** While this is part of the observation dictionary, standard `stable-baselines3` PPO with its default `MlpPolicy` typically **does not use this string directly** as input to the neural network for learning. The policy usually only processes the flattened `image` array and the `direction` integer. Advanced policies *can* be built to parse and use mission strings, but that's not the default setup in this project.

So, when the agent makes a decision, it's not just getting `(3,2)`. It's getting a small "picture" of its immediate surroundings, detailed by object types, colors, and states, plus its current heading. This allows for complex reactive behaviors like "If I see a locked red door, and I have a red key, and I am facing the door, then Toggle."

---

### Theoretical Distinction: State vs. Observation

You caught me being imprecise. In strict Reinforcement Learning theory, they are **NOT** the same.

Here is the exact distinction:

### 1. The State (God's View)
The **State ($S_t$)** represents the **complete, objective truth** of the entire universe at time $t$. If you knew the State, you would know absolutely everything.

*   **In your project (MiniGrid):** The State is the **entire** grid layout.
    *   It includes the `(x,y)` coordinates of the agent.
    *   It includes the `(x,y)` coordinates of the Key and the Door.
    *   It includes the map of every wall in the 5x5 room.
    *   It includes what the agent is holding.

If the agent had access to the **State**, it would know exactly where the key is, even if the key is behind a wall.

### 2. The Observation (Agent's View)
The **Observation ($O_t$)** is a **partial filter** of the state. It is the only data the agent actually receives.

*   **In your project:** The agent is **Partially Observable**.
    *   It **cannot** see the whole grid.
    *   It only sees that 7x7 square in front of its nose (the "image" I described earlier).
    *   It does **not** know its global `(x,y)` coordinates. It only knows what is immediately visible.

### Why I confused them (and why the code does too)
In practical code (like `stable-baselines3`), the mathematical function for the Policy is defined as $\pi(a|s)$ (Policy given State).

However, because we rarely give the agent the full God-View State (that would be cheating!), we feed the **Observation** into the function instead.

So, in your Python scripts:
*   **Variable Name:** `observation` (Correct)
*   **Math Concept:** Treated as $S$ (State) for the sake of the PPO algorithm.

### Summary
*   **State:** The Global Map + `(x,y)` coords. (The Agent **never** sees this).
*   **Observation:** The 7x7 pixel grid + `direction`. (This is what I described in the previous turn).

The "richness" I mentioned refers to the **Observation**. Even though it is a partial view, it is complex (colors, object types) compared to a simple "I am at x=3, y=5".

---

### Theoretical Implications: Feeding Observation vs. State

That is an excellent question. It hits the fundamental boundary between "Standard RL" and "Real-world RL."

Theoretically, moving from **State** to **Observation** changes the problem from a **Markov Decision Process (MDP)** to a **Partially Observable Markov Decision Process (POMDP)**.

Here are the three major theoretical shifts that happen:

### 1. The Loss of the "Markov Property" (The "Amnesia" Problem)
The core assumption of most RL algorithms (including PPO) is the **Markov Property**:
> "The future depends only on the present, not on the past."

*   **With State ($S_t$):** If you know the grid map, knowing where you were 5 minutes ago is irrelevant. You have all the info you need right now to make the perfect move.
*   **With Observation ($O_t$):** The Markov property breaks.
    *   *Example:* Imagine your agent is facing a white wall.
        *   Scenario A: It is at the start of the maze.
        *   Scenario B: It is at the end of the maze, but facing the wrong way.
    *   The observation (white wall) is **identical** in both cases.
    *   The correct action depends on *how it got there* (History). But since we only feed it the current $O_t$, PPO treats Scenario A and B as the exact same situation and will force the agent to take the exact same action in both, even if one leads to a win and one leads to death.

### 2. State Aliasing (Perceptual Blindness)
This is the formal term for the example above. Multiple different true **States** map to the same **Observation**.
$$ S_1 \neq S_2 \quad \text{but} \quad O(S_1) = O(S_2) $$

*   **Theoretical Impact:** The optimal policy is no longer deterministic.
    *   If you are in State 1, the best move is "Left".
    *   If you are in State 2, the best move is "Right".
    *   Since the agent only sees "White Wall" (which could be either), it cannot learn "Always go Left" or "Always go Right."
    *   It is forced to learn a **Stochastic Policy** (e.g., "50% Left, 50% Right") to hedge its bets, or a "safe" suboptimal policy (e.g., "Turn around").

### 3. The Need for Memory (Reactive vs. Recurrent)
Because the current observation $O_t$ is not enough to define the situation, the *true* optimal policy theoretically requires the **History of Observations**.

$$ \pi^*(a | o_t, o_{t-1}, ... o_0) $$

*   **What your code does (`MlpPolicy`):** It uses a Feed-Forward Network. It has **no memory**. It is a **Reactive Agent**. It decides solely on what it sees *right now*.
    *   *Result:* It acts like a goldfish. If it turns 360 degrees, it forgets what it saw at the start of the turn.
*   **What is theoretically required (`LstmPolicy`):** To solve a POMDP perfectly, you theoretically need a Recurrent Neural Network (LSTM or GRU). This gives the agent an internal "hidden state" (memory) that acts as a proxy for the missing True State.

Picked this specific environment because of sparse rewards.

## DQN Experiments (Feb 12, 2026)

### 1. The Kernel Size Mismatch
We encountered a `RuntimeError` when trying to use the standard `CnnPolicy` (NatureCNN) with MiniGrid's default grid observation.
*   **Reason:** MiniGrid outputs a 7x7 grid. The standard Atari CNN uses an 8x8 kernel in its first layer. You cannot fit an 8x8 kernel into a 7x7 input.
*   **Solution:** We used the `RGBImgPartialObsWrapper`. This converts the 7x7 grid into a **56x56 pixel image** (RGB). This is large enough for the standard CNN to process without crashing.

### 2. DQN on Pixels (FourRooms vs. Empty)
*   **MiniGrid-FourRooms-v0:** Failed. After 100k steps, the agent had 0 reward and spun in circles. Likely requires millions of steps due to sparse rewards and complex visual input.
*   **MiniGrid-Empty-8x8-v0:** **Success!** After 500k steps, the agent achieved a mean reward of **~0.96** and near-optimal pathing (12 steps).

### 3. Reproducing Results
To train the successful Pixel-DQN agent on Empty-8x8:
```bash
cd src
uv run prelim/qlearning/train_pixels.py
```

To watch the trained agent:
```bash
cd src
uv run prelim/qlearning/train_pixels.py --watch
```

## DQN on Pixels for MiniGrid-DoorKey-5x5-v0 (Feb 12, 2026)

**Objective:** Compare Baseline vs. Curriculum learning for DQN on `MiniGrid-DoorKey-5x5-v0` using pixel observations (56x56 RGB images) and the standard `CnnPolicy`.

**Training Setup:**
*   **Baseline Agent:** Trained for 500,000 steps directly on `MiniGrid-DoorKey-5x5-v0`.
    *   Command: `cd src && uv run prelim/keydqn/train_key.py`
*   **Curriculum Agent:** Trained for 500,000 steps in two stages:
    *   Stage 1: 100,000 steps on `MiniGrid-Empty-6x6-v0` (navigation).
    *   Stage 2: 400,000 steps on `MiniGrid-DoorKey-5x5-v0` (key/door mechanics), continuing from Stage 1 weights.
    *   Command: `cd src && uv run prelim/keydqn/train_cur.py`
*   **FrameStack Agent:** Trained for 500,000 steps directly on `MiniGrid-DoorKey-5x5-v0` with 4 frames stacked.
    *   Command: `cd src && uv run prelim/keydqn/train_key_framestack.py`

**Results after 500k steps:**
*   **Baseline Agent (Watched):** Consistently received **0.0 reward** and hit the 100-step limit in all watched episodes. It failed to reliably solve the `DoorKey-5x5-v0` task.
*   **Curriculum Agent (Watched):** Also consistently received **0.0 reward** and hit the 100-step limit in all watched episodes. It failed to reliably solve the `DoorKey-5x5-v0` task, despite promising training logs during Stage 2.
*   **FrameStack Agent (Watched):** **SUCCESS!** Consistently achieved high rewards (avg ~0.95) and low steps (avg ~13) per episode. This agent reliably solves `MiniGrid-DoorKey-5x5-v0`.

**Conclusion:**
Adding **FrameStack** (short-term memory) significantly improved the DQN agent's performance on `MiniGrid-DoorKey-5x5-v0`. This confirms that the partial observability (POMDP nature) of the environment, specifically the need for temporal context to understand sequences of actions like "turn left, move forward", was a critical factor. The Baseline and Curriculum agents (without FrameStack) were unable to consistently learn the complex sequence of actions required for the task within the given budget.

**Future Steps:**

1.  **Re-evaluate Curriculum with FrameStack:** Now that FrameStack is shown to be effective, apply it to the Curriculum learning approach for `DoorKey-5x5-v0` or even `DoorKey-8x8-v0`.
2.  **Increase Complexity:** Attempt to solve `MiniGrid-DoorKey-8x8-v0` with FrameStack.
3.  **Explore Longer-Term Memory:** Investigate `RecurrentPPO` (with LSTMs) as a next step for more complex, long-horizon POMDPs where short-term memory from FrameStack might not be sufficient.
4.  **Hyperparameter Tuning & Reward Shaping (if needed):** If challenges arise with more complex tasks, revisit these standard RL techniques.

---
**To watch the successful FrameStack agent on DoorKey-5x5:**
```bash
cd src
uv run prelim/keydqn/train_key_framestack.py --watch 100
```

**To watch the successful Pixel agent on Empty-8x8:**
```bash
cd src
uv run prelim/qlearning/train_pixels.py --watch 100
```

---

## Progress Report: Feb 13, 2026

### 1. Discussion on Reward Functions & Logging
*   **Observation:** Noticed that `rollout/ep_rew_mean` was missing from console logs during DQN training.
*   **Analysis:** This is typically due to episodes not completing within a logging interval or the need for an explicit monitor.
*   **Solution (Proposed, but not implemented due to user preference):** Using `stable_baselines3.common.monitor.Monitor` wrapper to ensure consistent logging of episode statistics.
*   **Discussion on Reward Shaping:** Explored mathematical formulations for reward shaping (distance-based potentials, curiosity-driven exploration, HER) to address sparse reward problems in MiniGrid.

### 2. Transition from DQN (Pixels) to PPO (Symbolic)
*   **The DQN Failure (Iteration 1):**
    *   **Setup:** `CnnPolicy` with `RGBImgPartialObsWrapper` (56x56x3 pixels).
    *   **Critical Failure:** Experienced severe instability and catastrophic forgetting. The agent would solve the maze during high exploration but "forget" the solution as exploration decayed.
    *   **Root Cause:** DQN's overestimation bias combined with an off-policy replay buffer led to **"buffer poisoning"** in this sparse reward setting. The agent's buffer filled with early failures or non-representative random successes, and the off-policy updates couldn't stabilize a robust policy.
*   **The PPO Solution (Iteration 2):**
    *   **Setup:** `MlpPolicy` with `FlatObsWrapper` (Symbolic 1D array).
    *   **Resolution:** PPO's **on-policy** Actor-Critic architecture and **gradient clipping** mechanically prevented catastrophic forgetting. By discarding old data and only training on the most recent policy's experience, it avoided the buffer poisoning issue.
    *   **Benefit:** Switching to symbolic observations removed the computer vision bottleneck, allowing lightning-fast convergence.
*   **Implementation:**
    *   Created `src/prelim/keydqn/train_dqn_flat_symbolic.py` (and later PPO scripts).
    *   Utilized `DummyVecEnv` to run 4 parallel environments for faster data collection.
    *   Added the `Monitor` wrapper to properly track and expose `ep_rew_mean` to TensorBoard.

### 3. Enhancements to PPO Curriculum Training & Watching
*   **Observation:** The existing PPO curriculum script (`src/prelim/ppoflat/train_ppo_cur.py`) only saved the final model, making it impossible to watch intermediate stages.
*   **Action:** Modified `src/prelim/ppoflat/train_ppo_cur.py` to save models after each stage:
    *   `Phase1_Model.zip` (after "Empty-5x5" training).
    *   `Phase2_Model.zip` (after "DoorKey-5x5" training, replacing "PPO_Curriculum_Master.zip").
*   **New Watcher Script:** Created `src/prelim/ppoflat/watch_curriculum.py`.
    *   Allows watching specific stages (1 or 2) by loading the corresponding model and environment.
    *   Example Usage: `uv run src/prelim/ppoflat/watch_curriculum.py 1`

### 4. PPO Parameter Explanation
### 5. Future Curriculum Directions
Based on the success of the PPO transfer learning (curriculum) approach, we have identified three potential tracks for future experiments:

1.  **Compositional Tool Use:**
    *   *Progression:* `MiniGrid-DoorKey-5x5-v0` $\to$ `MiniGrid-UnlockPickup-v0` $\to$ `MiniGrid-ObstructedMaze-Full-v0`
    *   *Goal:* Teach the agent to chain multiple interactions (find key, open door, move obstacle, pick up ball).

2.  **Procedural Generalization:**
    *   *Progression:* `MiniGrid-MultiRoom-N2-S4-v0` $\to$ `MiniGrid-MultiRoom-N4-S5-v0` $\to$ `MiniGrid-MultiRoom-N6-v0`
    *   *Goal:* Test the agent's ability to navigate increasingly large and complex generated maps without forgetting basic navigation skills.

3.  **Temporal Survival:**
    *   *Progression:* `MiniGrid-LavaGapS5-v0` $\to$ `MiniGrid-Dynamic-Obstacles-5x5-v0` $\to$ `MiniGrid-Dynamic-Obstacles-16x16-v0`
    *   *Goal:* Introduce dynamic elements and time pressure to force robust, safe policy learning.