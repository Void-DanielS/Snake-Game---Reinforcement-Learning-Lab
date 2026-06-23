import numpy as np
import random
from collections import defaultdict


class RLAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.01):
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min   = epsilon_min
        self.q_table       = defaultdict(lambda: np.zeros(4))

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        return int(np.argmax(self.q_table[state]))

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update(self, state, action, reward, next_state, next_action=None, done=False):
        raise NotImplementedError


class QLearningAgent(RLAgent):
    """Off-policy TD control. Target uses max Q over next state regardless of policy."""

    def update(self, state, action, reward, next_state, next_action=None, done=False):
        current_q = self.q_table[state][action]

        # Best possible Q value from next state (greedy target)
        max_next_q = 0.0 if done else np.max(self.q_table[next_state])

        # TD update: Q(s,a) <- Q(s,a) + alpha * [r + gamma * max Q(s',*) - Q(s,a)]
        td_target = reward + self.gamma * max_next_q
        self.q_table[state][action] += self.alpha * (td_target - current_q)


class SarsaAgent(RLAgent):
    """On-policy TD control. Target uses the action actually taken in next state."""

    def update(self, state, action, reward, next_state, next_action=None, done=False):
        current_q = self.q_table[state][action]

        # Q value of the actual next action taken (on-policy)
        next_q = 0.0 if done else self.q_table[next_state][next_action]

        # TD update: Q(s,a) <- Q(s,a) + alpha * [r + gamma * Q(s',a') - Q(s,a)]
        td_target = reward + self.gamma * next_q
        self.q_table[state][action] += self.alpha * (td_target - current_q)


class DynaQAgent(RLAgent):
    """Q-Learning + model-based planning. Replays n simulated experiences per step."""

    def __init__(self, planning_steps=10, **kwargs):
        super().__init__(**kwargs)
        self.planning_steps = planning_steps
        # Model stores (state, action) -> (reward, next_state, done)
        self.model          = {}
        self.visited        = []  # list of (state, action) pairs seen

    def update(self, state, action, reward, next_state, next_action=None, done=False):
        # --- Direct Q-Learning update ---
        current_q  = self.q_table[state][action]
        max_next_q = 0.0 if done else np.max(self.q_table[next_state])
        td_target  = reward + self.gamma * max_next_q
        self.q_table[state][action] += self.alpha * (td_target - current_q)

        # --- Model learning: store observed transition ---
        key = (state, action)
        if key not in self.model:
            self.visited.append(key)
        self.model[key] = (reward, next_state, done)

        # --- Planning: replay n random past experiences ---
        if len(self.visited) > 0:
            samples = random.choices(self.visited, k=min(self.planning_steps, len(self.visited)))
            for s, a in samples:
                r, ns, d = self.model[(s, a)]
                # Simulated Q-learning update from model
                sim_current_q  = self.q_table[s][a]
                sim_max_next_q = 0.0 if d else np.max(self.q_table[ns])
                sim_target     = r + self.gamma * sim_max_next_q
                self.q_table[s][a] += self.alpha * (sim_target - sim_current_q)
