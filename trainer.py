import numpy as np
import pickle
from collections import defaultdict
from environment import SnakeEnv
from agents import QLearningAgent, SarsaAgent, DynaQAgent

AGENT_NAMES = ["Q-Learning", "SARSA", "Dyna-Q"]


class TrainingSession:
    """Owns all mutable training state; UI reads from it, never writes to it."""

    def __init__(self, agents, envs, cfg):
        self.agents       = agents
        self.envs         = envs
        self.max_episodes = cfg["max_episodes"]
        self.max_steps    = cfg["max_steps"]

        self.states   = [e.reset() for e in envs]
        self.scores   = [0] * 3
        self.records  = [0] * 3
        self.episodes = [0] * 3
        self.ep_steps = [0] * 3
        self.actions  = [a.choose_action(s) for a, s in zip(agents, self.states)]

    @property
    def training_done(self) -> bool:
        return all(ep >= self.max_episodes for ep in self.episodes)

    def step(self):
        """Advance every agent by one environment step."""
        for i in range(3):
            if self.episodes[i] >= self.max_episodes:
                continue

            state  = self.states[i]
            action = self.actions[i]

            next_state, reward, done = self.envs[i].step(action)
            self.ep_steps[i] += 1

            if self.ep_steps[i] >= self.max_steps:
                done = True

            next_action = self.agents[i].choose_action(next_state)
            self.agents[i].update(state, action, reward, next_state,
                                  next_action=next_action, done=done)

            self.states[i]  = next_state
            self.actions[i] = next_action
            self.scores[i]  = self.envs[i].score

            if done:
                if self.envs[i].score > self.records[i]:
                    self.records[i] = self.envs[i].score
                self.episodes[i] += 1
                self.ep_steps[i]  = 0
                self.agents[i].decay_epsilon()
                self.states[i]  = self.envs[i].reset()
                self.scores[i]  = 0
                self.actions[i] = self.agents[i].choose_action(self.states[i])


class EvalSession:
    """Runs trained agents in greedy mode (no exploration, no Q-table updates)."""

    def __init__(self, agents, num_obstacles=0):
        self.agents = agents
        for a in self.agents:
            a.epsilon = 0.0

        self.envs      = [SnakeEnv(num_obstacles=num_obstacles) for _ in range(3)]
        self.states    = [e.reset() for e in self.envs]
        self.scores    = [0] * 3
        self.eval_ep   = [0] * 3
        self.eval_hist = [[] for _ in range(3)]

    def step(self):
        """Advance every agent by one greedy step."""
        for i in range(3):
            state  = self.states[i]
            action = int(np.argmax(self.agents[i].q_table[state]))
            next_state, reward, done = self.envs[i].step(action)
            self.states[i] = next_state
            self.scores[i] = self.envs[i].score
            if done:
                self.eval_hist[i].append(self.envs[i].score)
                self.eval_ep[i] += 1
                self.states[i]  = self.envs[i].reset()
                self.scores[i]  = 0


def create_agents(cfg: dict):
    """Instantiate all three RL agents from a hyperparameter config dict."""
    return [
        QLearningAgent(alpha=cfg["alpha"], gamma=cfg["gamma"],
                       epsilon=cfg["epsilon"], epsilon_decay=cfg["epsilon_decay"]),
        SarsaAgent    (alpha=cfg["alpha"], gamma=cfg["gamma"],
                       epsilon=cfg["epsilon"], epsilon_decay=cfg["epsilon_decay"]),
        DynaQAgent    (alpha=cfg["alpha"], gamma=cfg["gamma"],
                       epsilon=cfg["epsilon"], epsilon_decay=cfg["epsilon_decay"],
                       planning_steps=cfg["planning_steps"]),
    ]


def save_q_matrices(agents, filepath: str):
    """Write all three agents' Q-tables to a binary file."""
    data = {
        "agent_names": AGENT_NAMES,
        "q_tables": [dict(a.q_table) for a in agents],
        "configs": [
            {
                "alpha":       a.alpha,
                "gamma":       a.gamma,
                "epsilon_min": a.epsilon_min,
                **({"planning_steps": a.planning_steps} if hasattr(a, "planning_steps") else {}),
            }
            for a in agents
        ],
    }
    with open(filepath, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_q_matrices(filepath: str):
    """Read a saved Q-matrix file and return three agents ready for evaluation."""
    with open(filepath, "rb") as f:
        data = pickle.load(f)

    agent_classes = [QLearningAgent, SarsaAgent, DynaQAgent]
    agents_out = []

    for cls, cfg, q_dict in zip(agent_classes, data["configs"], data["q_tables"]):
        kw = {
            "alpha":         cfg["alpha"],
            "gamma":         cfg["gamma"],
            "epsilon":       0.0,
            "epsilon_decay": 1.0,
            "epsilon_min":   cfg.get("epsilon_min", 0.01),
        }
        if "planning_steps" in cfg:
            kw["planning_steps"] = cfg["planning_steps"]
        agent = cls(**kw)
        agent.q_table = defaultdict(lambda: np.zeros(4), q_dict)
        agents_out.append(agent)

    return agents_out


def load_q_matrices_for_transfer(filepath: str, cfg: dict):
    """Load Q-tables from file and return agents primed for continued training.

    Uses the hyperparameters from cfg (alpha, gamma, epsilon, etc.) so the
    agent resumes exploration from the configured epsilon rather than 0.
    """
    with open(filepath, "rb") as f:
        data = pickle.load(f)

    agent_classes = [QLearningAgent, SarsaAgent, DynaQAgent]
    agents_out = []

    for cls, saved_cfg, q_dict in zip(agent_classes, data["configs"], data["q_tables"]):
        kw = {
            "alpha":         cfg["alpha"],
            "gamma":         cfg["gamma"],
            "epsilon":       cfg["epsilon"],
            "epsilon_decay": cfg["epsilon_decay"],
            "epsilon_min":   saved_cfg.get("epsilon_min", 0.01),
        }
        if cls == DynaQAgent:
            kw["planning_steps"] = cfg["planning_steps"]
        agent = cls(**kw)
        agent.q_table = defaultdict(lambda: np.zeros(4), q_dict)
        agents_out.append(agent)

    return agents_out
