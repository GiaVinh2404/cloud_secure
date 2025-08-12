import os
import glob
import numpy as np
import pandas as pd
from pettingzoo.utils.agent_selector import agent_selector
from pettingzoo.utils.env import AECEnv
from gymnasium.spaces import Discrete, Box

class CloudSecurityEnv(AECEnv):
    metadata = {"render_modes": ["human"]}

    def __init__(self, data_dir="data/processed", attack_ratio=0.3 , max_steps=1000):
        super().__init__()
        self.attack_ratio = attack_ratio
        self.max_steps = max_steps

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(base_dir, data_dir, "*.parquet")
        all_files = glob.glob(data_path)

        if not all_files:
            raise FileNotFoundError(f"Không tìm thấy file parquet ở {data_path}")

        df_list = [pd.read_parquet(f) for f in all_files]
        df = pd.concat(df_list, ignore_index=True)

        # Chuẩn hóa label: 0 cho Benign, 1 cho Attack
        df["Label"] = df["Label"].replace({"BENIGN": "Benign", "Benign": 0}).apply(lambda x: 0 if x == 0 or x == "Benign" else 1)

        self.feature_cols = [
            "Flow Duration",
            "Total Fwd Packets", "Total Backward Packets",
            "Fwd Packet Length Min", "Fwd Packet Length Mean", "Bwd Packet Length Mean",
            "Flow IAT Mean", "Flow IAT Std", "Fwd IAT Total", "Bwd IAT Total"
        ]

        # Kiểm tra cột thiếu
        missing_cols = [col for col in self.feature_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Thiếu các cột đặc trưng: {missing_cols}")

        features = df[self.feature_cols].fillna(0).astype(np.float32).values
        self.min_vals = features.min(axis=0)
        self.max_vals = features.max(axis=0)
        self.base_features = (features - self.min_vals) / (self.max_vals - self.min_vals + 1e-9)
        self.base_labels = df["Label"].values.astype(np.int32)

        self.possible_agents = ["predictor_agent", "action_agent"]
        self.agent_selector = agent_selector(self.possible_agents)

        self.observation_spaces = {
            "predictor_agent": Box(low=0, high=1, shape=(len(self.feature_cols),), dtype=np.float32),
            "action_agent": Box(low=0, high=1, shape=(len(self.feature_cols) + 1,), dtype=np.float32)
        }
        self.action_spaces = {
            "predictor_agent": Discrete(2),
            "action_agent": Discrete(2)
        }

    def reset(self, seed=None, options=None):
        # Khởi tạo lại môi trường
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(self.base_features))
        self.features = self.base_features[perm]
        self.labels = self.base_labels[perm]

        if self.attack_ratio is not None:
            mask = rng.random(len(self.features)) < self.attack_ratio
            self.labels = np.where(mask, 1, 0)

        self.agents = self.possible_agents[:]
        self.agent_selector.reinit(self.agents)
        self.agent_selection = self.agent_selector.next()

        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.last_prediction = None
        self.idx = 0
        self.step_count = 0

        # Reset streak tracking
        self.correct_streak = {agent: 0 for agent in self.possible_agents}
        self.last_actions = {agent: None for agent in self.possible_agents}

    def observe(self, agent):
        if self.idx >= len(self.features):
            return None
        if agent == "predictor_agent":
            return self.features[self.idx]
        elif agent == "action_agent":
            pred_info = np.array([self.last_prediction if self.last_prediction is not None else 0], dtype=np.float32)
            return np.concatenate([self.features[self.idx], pred_info])
        else:
            raise ValueError(f"Unknown agent: {agent}")

    def step(self, action):
        current_agent = self.agent_selection
        label = self.labels[self.idx]
        reward = 0

        # Hàm tính bonus reward theo streak
        def get_streak_bonus(streak):
            if streak >= 10:
                return 2.0
            elif streak >= 5:
                return 1.0
            elif streak >= 3:
                return 0.5
            return 0

        if current_agent == "predictor_agent":
            self.last_prediction = action
            # Reward chính
            if action == 1 and label == 1:
                reward = 2.0  # TP (tăng thưởng)
            elif action == 0 and label == 0:
                reward = 1.2  # TN (tăng nhẹ)
            elif action == 1 and label == 0:
                reward = -0.7 # FP (tăng phạt)
            elif action == 0 and label == 1:
                reward = -2.5 # FN (tăng phạt mạnh)

            # Cập nhật streak
            if (action == label):
                self.correct_streak[current_agent] += 1
            else:
                self.correct_streak[current_agent] = 0

            # Thêm bonus reward theo streak
            bonus = get_streak_bonus(self.correct_streak[current_agent])
            reward += bonus

            # Penalty nếu hành động lặp lại nhiều lần (ví dụ 3 bước liền)
            if self.last_actions[current_agent] == action:
                self.correct_streak[current_agent] = max(self.correct_streak[current_agent] - 0.2, 0)
                reward -= 0.1  # phạt nhẹ hành động lặp lại

            self.last_actions[current_agent] = action
            self._cumulative_rewards[current_agent] += reward

        elif current_agent == "action_agent":
            pred = self.last_prediction
            # Reward chính
            if pred == 1 and action == 1:
                reward = 2.5   # Xác nhận tấn công (tăng thưởng)
            elif pred == 1 and action == 0:
                reward = -3.5  # Bỏ qua tấn công (tăng phạt)
            elif pred == 0 and action == 0:
                reward = 1.2   # Cho qua benign đúng (tăng nhẹ)
            else:
                reward = -1.2  # Chặn nhầm benign (tăng phạt)

            # Cập nhật streak
            correct_action = ((pred == 1 and action == 1) or (pred == 0 and action == 0))
            if correct_action:
                self.correct_streak[current_agent] += 1
            else:
                self.correct_streak[current_agent] = 0

            # Bonus reward
            bonus = get_streak_bonus(self.correct_streak[current_agent])
            reward += bonus

            # Phạt nếu action lặp lại nhiều lần
            if self.last_actions[current_agent] == action:
                self.correct_streak[current_agent] = max(self.correct_streak[current_agent] - 0.2, 0)
                reward -= 0.1

            self.last_actions[current_agent] = action
            self._cumulative_rewards[current_agent] += reward

            # Chuyển sang mẫu tiếp theo
            self.idx += 1
            self.step_count += 1
            if self.idx >= len(self.features) or self.step_count >= self.max_steps:
                self.terminations = {agent: True for agent in self.agents}
                self.agents = []

        self.rewards[current_agent] = reward
        self.agent_selection = self.agent_selector.next()
        self._was_last_step = True

    def render(self, mode="human"):
        if self.agent_selection is None or self.idx >= len(self.features):
            return
        obs_to_render = self.observe(self.agent_selection)
        print(f"Step: {self.idx}, Agent: {self.agent_selection}, Observation shape: {obs_to_render.shape if obs_to_render is not None else None}, Rewards: {self.rewards}")

    def close(self):
        pass

if __name__ == "__main__":
    env = CloudSecurityEnv()
    env.reset()

    while env.agents:
        agent = env.agent_selection
        observation = env.observe(agent)
        action = env.action_spaces[agent].sample()
        env.step(action)
        env.render()
    
    print("\nEnvironment finished.")
    print("Final rewards:", env._cumulative_rewards)