# Snake RL — Q-Learning vs SARSA vs Dyna-Q

A visual reinforcement learning trainer that runs three classic tabular agents side-by-side on a Snake grid. Watch them learn in real time, compare their performance, add obstacles to increase difficulty, and transfer pre-trained Q-tables to new environments.

---

## Features

- **Three agents trained simultaneously** — Q-Learning, SARSA, and Dyna-Q share the same hyperparameters so the only variable is the algorithm.
- **Live visualisation** — resizable pygame window with per-agent grids, live stats, leaderboard sidebar, and speed control (1× – 32×).
- **Configurable obstacles** — place N random wall tiles on the grid; the snake dies on contact, forcing agents to navigate around them.
- **Q-matrix persistence** — save trained Q-tables to a `.pkl` file and reload them later for evaluation or as a starting point for further training.
- **Transfer learning** — load a Q-table trained in one setting (e.g. no obstacles) and continue training in a harder setting (e.g. with obstacles).

---

## Project Structure

```
Snake-Agent/
├── main.py          # Pygame UI — Config, Training, and Evaluation screens
├── trainer.py       # Backend — training/eval sessions, Q-matrix save & load
├── agents.py        # RL agent classes (QLearningAgent, SarsaAgent, DynaQAgent)
├── environment.py   # Snake game environment (state, step, reward, obstacles)
├── requirements.txt # Runtime dependencies
└── .gitignore
```

---

## Requirements

- Python **3.10** or newer (uses `tuple[…]` type hints)
- `pygame >= 2.5.0`
- `numpy >= 1.24.0`

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/Snake-Agent.git
cd Snake-Agent
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
```

Activate it:

| Platform | Command |
|----------|---------|
| Windows (PowerShell) | `.\venv\Scripts\Activate.ps1` |
| Windows (CMD) | `venv\Scripts\activate.bat` |
| macOS / Linux | `source venv/bin/activate` |

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running

```bash
python main.py
```

The application opens a resizable window (default 1400 × 820) and walks you through three screens.

---

## Screen-by-Screen Guide

### Config Screen

Tune hyperparameters before training.

#### Hyperparameters (left column)

| Field | Key | Default | Effect |
|-------|-----|---------|--------|
| Learning Rate (α) | `alpha` | `0.1` | Step size for Q-table updates. Higher = faster but less stable. |
| Discount Factor (γ) | `gamma` | `0.9` | Weight given to future rewards. Values close to 1 make the agent plan further ahead. |
| Initial Epsilon (ε) | `epsilon` | `1.0` | Starting exploration rate. `1.0` = fully random. |
| Epsilon Decay | `epsilon_decay` | `0.995` | Multiplied by ε after each episode. Lower = exploration shrinks faster. |
| Dyna-Q Planning Steps | `planning_steps` | `10` | Simulated transitions Dyna-Q replays per real step. Higher = faster learning, more CPU. |

#### Training Config (right column)

| Field | Key | Default | Effect |
|-------|-----|---------|--------|
| Max Training Episodes | `max_episodes` | `3000` | Training stops when every agent reaches this count. |
| Max Steps / Episode | `max_steps` | `500` | Episode ends early after this many steps to prevent infinite loops. |
| Number of Obstacles | `num_obstacles` | `0` | Random wall tiles added to the grid. The snake dies on contact. |

#### Buttons

| Button | Action |
|--------|--------|
| **▶ START TRAINING** | Validate the form and begin training with fresh agents. |
| **⬆ LOAD (EVAL)** | Pick a `.pkl` file and go straight to the Evaluation screen — no training needed. |
| **⟳ TRANSFER LEARN** | Pick a `.pkl` file, load its Q-tables into new agents at the configured epsilon, and start training from that prior knowledge. |

Click a field to select it, type a number, then press **Tab** or **Enter** to advance.

---

### Training Screen

Three Snake grids run simultaneously, one per algorithm.

- **Game panels** — live snake, food, obstacles, and a stats strip (episode, score, best, ε).
- **Progress bar** (header) — tracks the slowest agent toward the episode target.
- **Leaderboard sidebar** — ranks agents by best score, updated in real time.

#### Controls

| Key | Action |
|-----|--------|
| `Space` | Pause / resume |
| `+` / `=` | Increase speed (1× → 2× → 4× → 8× → 16× → 32×) |
| `-` | Decrease speed |
| `Esc` | Abort and return to Config screen |

#### Training Complete overlay

| Key | Action |
|-----|--------|
| `Space` | Proceed to Evaluation screen |
| `S` | Save all three Q-tables to a `.pkl` file |
| `Esc` | Return to Config screen |

---

### Evaluation Screen

Runs trained agents in **pure greedy mode** (ε = 0, no Q-table updates).

- **Game panels** — live greedy play with score, best, running average, and a sparkline of the last 10 episodes.
- **Results sidebar** — ranks agents by average score.

#### Controls

| Key | Action |
|-----|--------|
| `Esc` | Return to Config screen |

---

## Obstacles

When **Number of Obstacles** is set to a value greater than 0, each environment instance places that many random wall tiles on the grid at startup. Obstacles:

- Are fixed for the lifetime of the environment (do not move between episodes).
- Kill the snake on contact (same penalty as hitting a wall: −10, episode ends).
- Are excluded from food placement, so food always spawns on a reachable cell.
- Do **not** expand the state space — the three danger bits (`danger_front`, `danger_left`, `danger_right`) already encode obstacle proximity the same way they encode walls and body segments.

---

## Q-Matrix Save & Load

### Saving

1. Train until the **Training Complete** overlay appears.
2. Press **`S`** and choose a location in the native save dialog.
3. A `.pkl` file is written containing each agent's full Q-table and its hyperparameters.

### Loading for Evaluation

1. On the Config screen click **⬆ LOAD (EVAL)**.
2. Select a `.pkl` file.
3. The application enters the Evaluation screen immediately with ε = 0.

### Loading for Transfer Learning

1. On the Config screen, set the hyperparameters (including **Number of Obstacles**) for the *new* training run.
2. Click **⟳ TRANSFER LEARN** and select a `.pkl` file.
3. The saved Q-tables are loaded into fresh agents at the configured epsilon, then training begins as normal.

The pre-trained knowledge transfers because the 8-element state tuple is the same regardless of the source environment — the danger bits capture walls, body segments, and obstacles uniformly. An agent trained on an open grid already knows "avoid danger in front"; it just needs to refine that policy for the denser obstacle layout.

> **Compatibility note:** Q-table files are Python pickle objects. They are portable across OS but may break if the state representation in `environment.py` changes (i.e., if the tuple structure is modified).

---

## State & Reward Design

Each game state is an 8-element tuple:

```
(danger_front, danger_left, danger_right, direction, food_up, food_down, food_left, food_right)
```

- `danger_*` — 1 if the cell in that relative direction is a wall, body segment, or obstacle; 0 otherwise.
- `direction` — current heading (0 UP · 1 DOWN · 2 LEFT · 3 RIGHT).
- `food_*` — 1 if food is in that absolute direction from the head.

This yields at most **512 unique states**, making tabular Q-learning practical without function approximation.

| Event | Reward |
|-------|--------|
| Eating food | +10 |
| Hitting a wall, body segment, or obstacle | −10 (episode ends) |
| Each step without food | −0.1 |
| Starvation timeout (`grid² × 2` steps) | episode ends |

---

## Algorithm Comparison

| | Q-Learning | SARSA | Dyna-Q |
|---|---|---|---|
| **Type** | Off-policy TD | On-policy TD | Model-based |
| **Target** | max Q(s', ·) | Q(s', a') | max Q(s', ·) + planning |
| **Exploration effect on learning** | Ignored in update | Included in update | Ignored (like Q-Learning) |
| **Sample efficiency** | Moderate | Moderate | High |
| **CPU per step** | Low | Low | Low + planning_steps |

Dyna-Q typically converges in roughly half the episodes of the other two because each real transition triggers *n* additional simulated updates from the learned model.

---

## Tips

- **3 000 episodes** is usually enough for all three agents to converge on a 10 × 10 open grid.
- **Increase `planning_steps`** (e.g. 20 – 50) to make Dyna-Q learn noticeably faster, at the cost of higher CPU usage.
- **Lower `epsilon_decay`** (e.g. 0.99) to reduce exploration faster; raise it (e.g. 0.999) to keep the agent exploring longer.
- Use **32× speed** for long training runs, then drop to 1× to watch the learned policy in action.
- Start with **0 obstacles**, save the Q-matrices, then use **⟳ TRANSFER LEARN** with 5 – 10 obstacles and a lower epsilon (e.g. 0.3) to see how quickly the agents adapt.
- With many obstacles the starvation timeout may trigger often early in training — consider increasing **Max Steps / Episode** to give agents more time to find food.
