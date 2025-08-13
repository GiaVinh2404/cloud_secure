import numpy as np
import random
import pandas as pd
from gymnasium.spaces import Discrete, Box
from pettingzoo.utils import agent_selector
from pettingzoo.utils.env import AECEnv

class CloudSecurityEnv(AECEnv):
    metadata = {"render_modes": ["human"]}

    def __init__(self, dataset_path=None, max_steps=1000):
        super().__init__()
        self.max_steps = max_steps

        # Load dataset từ CSV
        if dataset_path:
            df = pd.read_csv(dataset_path)
            # Giả sử cột cuối cùng là "Label" với 0=normal, 1=malicious
            self.dataset = [
                {"features": row[:-1].values.astype(np.float32), "label": int(row[-1])}
                for _, row in df.iterrows()
            ]
        else:
            self.dataset = self._generate_dummy_data()

        # Loại request: 0 = normal, 1 = malicious
        self.attack_types = {0: "normal", 1: "malicious"}

        # Agents
        self.possible_agents = ["attacker_agent", "predictor_agent", "action_agent"]
        self.agent_selector = agent_selector.AgentSelector(self.possible_agents)

        # Observation spaces
        feature_dim = len(self.dataset[0]["features"])
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
            # One-hot vector: attacker chỉ thấy loại request đã gửi
            obs = np.zeros(2, dtype=np.float32)
            obs[self.current_label] = 1.0
            return obs

        elif agent == "predictor_agent":
            # Nếu chưa có sample, trả về zeros
            if self.current_sample is None:
                return np.zeros_like(self.dataset[0]["features"], dtype=np.float32)
            return self.current_sample["features"]

        elif agent == "action_agent":
            # Thông tin prediction gần nhất
            pred_info = np.array([self.last_prediction if self.last_prediction is not None else 0], dtype=np.float32)

            # Nếu chưa có sample, trả về zeros cho phần features
            if self.current_sample is None:
                obs_features = np.zeros(self.observation_spaces["action_agent"].shape[0]-1, dtype=np.float32)
            else:
                obs_features = self.current_sample["features"]

            return np.concatenate([obs_features, pred_info])

        else:
            raise ValueError(f"Unknown agent: {agent}")

    def step(self, action):
        current_agent = self.agent_selection
        reward = 0

        if current_agent == "attacker_agent":
            # Chọn loại request
            label_choice = action
            candidates = [d for d in self.dataset if d["label"] == label_choice]
            self.current_sample = random.choice(candidates)
            self.current_label = self.current_sample["label"]

        elif current_agent == "predictor_agent":
            pred = action
            self.last_prediction = pred
            if pred == 1 and self.current_label == 1:  # detect malicious
                reward = 2.0
            elif pred == 0 and self.current_label == 0:  # detect normal
                reward = 1.0
            elif pred == 1 and self.current_label == 0:  # false positive
                reward = -0.7
            elif pred == 0 and self.current_label == 1:  # miss malicious
                reward = -2.0

        elif current_agent == "action_agent":
            act = action
            pred = self.last_prediction if self.last_prediction is not None else 0

            if pred == 1 and act == 1:  # correct block
                reward = 2.5
            elif pred == 1 and act == 0:  # detect attack but allow
                reward = -3.5
            elif pred == 0 and act == 0:  # allow normal
                reward = 1.2
            else:  # allow attack or block normal
                reward = -1.2

            # Attacker reward: tấn công qua mặt
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
