import numpy as np
import random
import pandas as pd
from gymnasium.spaces import Discrete, Box
from pettingzoo.utils.agent_selector import agent_selector
from pettingzoo.utils.env import AECEnv
import os

class CloudSecurityEnv(AECEnv):
    metadata = {"render_modes": ["human"]}

    def __init__(self, dataset_path=None, max_steps=1000):
        super().__init__()
        self.max_steps = max_steps

        # Load dataset từ Parquet hoặc CSV, chỉ giữ DataFrame
        if dataset_path:
            ext = os.path.splitext(dataset_path)[-1].lower()
            if ext == ".parquet":
                df = pd.read_parquet(dataset_path)
            elif ext == ".csv":
                df = pd.read_csv(dataset_path)
            else:
                raise ValueError("Unsupported file format for dataset_path")

            feature_cols = [c for c in df.columns if c.lower() != "label"]
            self.df = df
            self.feature_cols = feature_cols
        else:
            # Dummy data
            self.df = pd.DataFrame(
                np.random.rand(20, 10), columns=[f"f{i}" for i in range(10)]
            )
            self.df["Label"] = [0]*10 + [1]*10
            self.feature_cols = [f"f{i}" for i in range(10)]

        # Loại request: 0 = normal, 1 = malicious
        self.attack_types = {0: "normal", 1: "malicious"}

        # Agents
        self.possible_agents = ["attacker_agent", "predictor_agent", "action_agent"]
        self.agent_selector = agent_selector(self.possible_agents)

        # Observation spaces
        feature_dim = len(self.feature_cols)
        self.observation_spaces = {
            "attacker_agent": Box(low=0, high=1, shape=(2,), dtype=np.float32),
            "predictor_agent": Box(low=-np.inf, high=np.inf, shape=(feature_dim,), dtype=np.float32),
            "action_agent": Box(low=-np.inf, high=np.inf, shape=(feature_dim+1,), dtype=np.float32),
        }

        # Action spaces
        self.action_spaces = {
            "attacker_agent": Discrete(2),  # 0=normal, 1=malicious
            "predictor_agent": Discrete(2),  # 0=benign, 1=attack
            "action_agent": Discrete(2),     # 0=allow, 1=block
        }

        # State variables
        self.current_sample = None
        self.current_label = 0
        self.last_prediction = None
        self.step_count = 0

    def _generate_dummy_data(self):
        benign = [np.random.rand(10) for _ in range(10)]
        malicious = [np.random.rand(10) + 0.5 for _ in range(10)]
        data = [{"features": x, "label": 0} for x in benign]
        data += [{"features": x, "label": 1} for x in malicious]
        return data

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.agent_selector.reinit(self.agents)
        self.agent_selection = self.agent_selector.next()

        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self.last_prediction = None
        self.step_count = 0
        self.current_sample = None
        self.current_label = 0

    def observe(self, agent):
        if agent == "attacker_agent":
            obs = np.zeros(2, dtype=np.float32)
            obs[self.current_label] = 1.0
            return obs

        elif agent == "predictor_agent":
            if self.current_sample is None:
                return np.zeros(len(self.feature_cols), dtype=np.float32)
            return self.current_sample["features"]

        elif agent == "action_agent":
            pred_info = np.array([self.last_prediction if self.last_prediction is not None else 0], dtype=np.float32)
            if self.current_sample is None:
                obs_features = np.zeros(len(self.feature_cols), dtype=np.float32)
            else:
                obs_features = self.current_sample["features"]
            return np.concatenate([obs_features, pred_info])

        else:
            raise ValueError(f"Unknown agent: {agent}")

    def step(self, action):
        current_agent = self.agent_selection
        reward = 0

        if current_agent == "attacker_agent":
            label_choice = action
            candidates = self.df[self.df["Label"] == label_choice]
            row = candidates.sample(1).iloc[0]
            self.current_sample = {
                "features": row[self.feature_cols].values.astype(np.float32),
                "label": int(row["Label"])
            }
            self.current_label = self.current_sample["label"]

        elif current_agent == "predictor_agent":
            pred = action
            self.last_prediction = pred
            if pred == 1 and self.current_label == 1:
                reward = 2.0
            elif pred == 0 and self.current_label == 0:
                reward = 1.0
            elif pred == 1 and self.current_label == 0:
                reward = -0.7
            elif pred == 0 and self.current_label == 1:
                reward = -2.0

        elif current_agent == "action_agent":
            act = action
            pred = self.last_prediction if self.last_prediction is not None else 0

            if pred == 1 and act == 1:
                reward = 2.5
            elif pred == 1 and act == 0:
                reward = -3.5
            elif pred == 0 and act == 0:
                reward = 1.2
            else:
                reward = -1.2

            if self.current_label == 1 and act == 0:
                self.rewards["attacker_agent"] = 3.0
            elif self.current_label == 1 and act == 1:
                self.rewards["attacker_agent"] = -1.0
            else:
                self.rewards["attacker_agent"] = 0

            self.step_count += 1
            if self.step_count >= self.max_steps:
                self.terminations = {agent: True for agent in self.agents}
                self.agents = []

        self.rewards[current_agent] = reward
        self._cumulative_rewards[current_agent] += reward
        self.agent_selection = self.agent_selector.next()

    def render(self, mode="human"):
        print(f"Agent: {self.agent_selection}, Rewards: {self.rewards}, Label: {self.current_label}")

    def close(self):
        pass