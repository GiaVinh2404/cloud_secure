import numpy as np
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

        # Load dataset
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
            # dummy dataset
            self.df = pd.DataFrame(
                np.random.rand(20, 10), columns=[f"f{i}" for i in range(10)]
            )
            self.df["Label"] = [0] * 10 + [1] * 10
            self.feature_cols = [f"f{i}" for i in range(10)]

        self.attack_types = {0: "normal", 1: "malicious"}

        self.possible_agents = ["attacker_agent", "predictor_agent", "action_agent"]
        self.agent_selector = agent_selector(self.possible_agents)

        feature_dim = len(self.feature_cols)
        self.observation_spaces = {
            "attacker_agent": Box(low=0, high=1, shape=(2,), dtype=np.float32),
            "predictor_agent": Box(low=-np.inf, high=np.inf, shape=(feature_dim,), dtype=np.float32),
            "action_agent": Box(low=-np.inf, high=np.inf, shape=(feature_dim + 1,), dtype=np.float32),
        }

        self.action_spaces = {
            "attacker_agent": Discrete(2),
            "predictor_agent": Discrete(2),
            "action_agent": Discrete(2),
        }

        self.current_sample = None
        self.current_label = 0
        self.last_prediction = None
        self.step_count = 0
        self.predictor_action = None  # Thêm biến để lưu hành động của predictor

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.agent_selector.reinit(self.agents)
        self.agent_selection = self.agent_selector.next()

        # reset env state
        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self.last_prediction = None
        self.step_count = 0
        self.current_sample = None
        self.current_label = 0

        observations = {agent: self.observe(agent) for agent in self.agents}
        return observations, self.infos

    def observe(self, agent):
        if agent == "attacker_agent":
            obs = np.zeros(2, dtype=np.float32)
            if self.current_label is not None:
                obs[self.current_label] = 1.0
            return obs

        elif agent == "predictor_agent":
            if self.current_sample is None:
                return np.zeros(len(self.feature_cols), dtype=np.float32)
            return self.current_sample["features"]

        elif agent == "action_agent":
            pred_info = np.array(
                [self.last_prediction if self.last_prediction is not None else 0],
                dtype=np.float32,
            )
            if self.current_sample is None:
                obs_features = np.zeros(len(self.feature_cols), dtype=np.float32)
            else:
                obs_features = self.current_sample["features"]
            return np.concatenate([obs_features, pred_info])

        else:
            raise ValueError(f"Unknown agent: {agent}")

    def step(self, action):
        current_agent = self.agent_selection

        if self.terminations[current_agent] or self.truncations[current_agent]:
            self.agent_selection = self.agent_selector.next()
            return
        
        # Reset rewards for this step to zero for all agents before calculating new rewards
        self.rewards = {agent: 0.0 for agent in self.agents}

        if current_agent == "attacker_agent":
            label_choice = action
            candidates = self.df[self.df["Label"] == label_choice]
            if candidates.empty:
                row = self.df.sample(1).iloc[0]
            else:
                row = candidates.sample(1).iloc[0]
            self.current_sample = {
                "features": row[self.feature_cols].values.astype(np.float32),
                "label": int(row["Label"]),
            }
            self.current_label = self.current_sample["label"]

        elif current_agent == "predictor_agent":
            self.last_prediction = action
            # The predictor's reward is based solely on its prediction accuracy.
            if self.last_prediction == self.current_label:
                self.rewards[current_agent] = 1.0
            else:
                self.rewards[current_agent] = -1.0
        
        elif current_agent == "action_agent":
            action_decision = action
            pred = self.last_prediction if self.last_prediction is not None else 0
            label = self.current_label

            # Case 1: Phòng thủ thành công (True Positive)
            if label == 1 and pred == 1 and action_decision == 1:
                self.rewards["attacker_agent"] -= 2.0  # Phạt attacker
                self.rewards[current_agent] = 3.0       # Thưởng lớn cho action_agent
                
            # Case 2: Phát hiện sai (False Positive)
            elif label == 0 and pred == 1 and action_decision == 1:
                self.rewards[current_agent] = -1.0      # Phạt action_agent vì hành động sai
                
            # Case 3: Bỏ lọt tấn công (False Negative)
            elif label == 1 and pred == 0:
                self.rewards["attacker_agent"] = 2.0   # Thưởng lớn cho attacker
                self.rewards["predictor_agent"] = -2.0 # Phạt nặng predictor
                self.rewards[current_agent] = -2.0      # Phạt action_agent vì không thể hành động
                
            # Case 4: Mẫu bình thường và xử lý đúng (True Negative)
            elif label == 0 and pred == 0 and action_decision == 0:
                self.rewards[current_agent] = 1.0       # Thưởng cho action_agent vì hành động đúng
            
            # Case 5: Bỏ lọt tấn công khi predictor dự đoán đúng (FN-Action)
            elif label == 1 and pred == 1 and action_decision == 0:
                self.rewards["attacker_agent"] = 1.0     # Thưởng attacker
                self.rewards[current_agent] = -2.0      # Phạt nặng action_agent

            # Increment step count and check for termination/truncation
            self.step_count += 1
            if self.step_count >= self.max_steps:
                self.truncations = {agent: True for agent in self.agents}
                self.agents = []

        self._accumulate_rewards()
        self.agent_selection = self.agent_selector.next()

    def render(self, mode="human"):
        print(
            f"Agent: {self.agent_selection}, Rewards: {self.rewards}, Label: {self.current_label}"
        )

    def close(self):
        pass