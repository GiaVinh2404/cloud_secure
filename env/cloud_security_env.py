import numpy as np
from gymnasium.spaces import Discrete, Box
from pettingzoo.utils import agent_selector
from pettingzoo.utils.env import AECEnv

class CloudSecurityEnv(AECEnv):
    metadata = {"render_modes": ["human"]}

    def __init__(self, max_steps=1000):
        super().__init__()
        self.max_steps = max_steps

        self.attack_types = {
            0: "benign",
            1: "sql_injection",
            2: "xss_attack",
            3: "dos_attack",
            # có thể mở rộng thêm
        }

        self.possible_agents = ["attacker_agent", "predictor_agent", "action_agent"]
        self.agent_selector = agent_selector.AgentSelector(self.possible_agents)

        # observation space attacker: có thể là một vector đại diện kiểu tấn công hoặc một scalar loại tấn công
        self.observation_spaces = {
            "attacker_agent": Box(low=0, high=1, shape=(len(self.attack_types),), dtype=np.float32),
            "predictor_agent": Box(low=0, high=1, shape=(10,), dtype=np.float32),  # giả định feature vector 10 chiều
            "action_agent": Box(low=0, high=1, shape=(11,), dtype=np.float32),  # thêm 1 cho prediction info
        }
        # attacker chọn loại tấn công từ 0 đến 3 (4 loại)
        self.action_spaces = {
            "attacker_agent": Discrete(len(self.attack_types)),
            "predictor_agent": Discrete(2),
            "action_agent": Discrete(2)
        }

        self.current_request_vector = None
        self.current_label = 0  # 0 benign, >0 attack_type

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

        # Tạo request đầu tiên mặc định benign
        self.current_request_vector = np.zeros(len(self.attack_types), dtype=np.float32)
        self.current_label = 0

    def observe(self, agent):
        if agent == "attacker_agent":
            # attacker nhìn vector tấn công hiện tại (one-hot)
            return self.current_request_vector
        elif agent == "predictor_agent":
            # predictor nhận vector đặc trưng mô phỏng từ attacker tạo
            # Ví dụ convert attack type sang vector đặc trưng (giả lập)
            base_vector = np.random.rand(10) * 0.1  # baseline benign noise
            if self.current_label > 0:
                base_vector += np.eye(10)[self.current_label % 10] * 0.8  # tạo đặc trưng attack kiểu tương ứng
            return base_vector.astype(np.float32)
        elif agent == "action_agent":
            pred_info = np.array([self.last_prediction if self.last_prediction is not None else 0], dtype=np.float32)
            predictor_obs = self.observe("predictor_agent")
            return np.concatenate([predictor_obs, pred_info])
        else:
            raise ValueError(f"Unknown agent: {agent}")

    def step(self, action):
        current_agent = self.agent_selection
        reward = 0

        if current_agent == "attacker_agent":
            # attacker chọn loại tấn công
            attack_type = action
            self.current_request_vector = np.zeros(len(self.attack_types), dtype=np.float32)
            self.current_request_vector[attack_type] = 1.0
            self.current_label = attack_type
            reward = 0  # reward tính ở bước action_agent
        elif current_agent == "predictor_agent":
            # predictor dự đoán attack(1) hoặc benign(0)
            pred = action
            self.last_prediction = pred
            # reward cho predictor như bình thường (tuỳ thiết kế)
            if pred == 1 and self.current_label > 0:
                reward = 2.0
            elif pred == 0 and self.current_label == 0:
                reward = 1.0
            elif pred == 1 and self.current_label == 0:
                reward = -0.7
            elif pred == 0 and self.current_label > 0:
                reward = -2.0
            else:
                reward = 0
        elif current_agent == "action_agent":
            # action agent quyết định block(1) hoặc allow(0)
            act = action
            pred = self.last_prediction if self.last_prediction is not None else 0

            # reward cho action agent (vd như trước)
            if pred == 1 and act == 1:
                reward = 2.5
            elif pred == 1 and act == 0:
                reward = -3.5
            elif pred == 0 and act == 0:
                reward = 1.2
            else:
                reward = -1.2

            # reward cho attacker theo hiệu quả tấn công
            if self.current_label > 0:  # attack
                if act == 0:  # cho phép tấn công thành công
                    self.rewards["attacker_agent"] = 3.0  # thưởng lớn
                else:  # block tấn công
                    self.rewards["attacker_agent"] = -1.0  # phạt attacker
            else:
                self.rewards["attacker_agent"] = 0  # benign không thưởng phạt

            self.step_count += 1

            if self.step_count >= self.max_steps:
                self.terminations = {agent: True for agent in self.agents}
                self.agents = []

        self.rewards[current_agent] = reward
        self._cumulative_rewards[current_agent] += reward
        self.agent_selection = self.agent_selector.next()
        self._was_last_step = True

    def render(self, mode="human"):
        if self.agent_selection is None:
            return
        print(f"Agent: {self.agent_selection}, Reward: {self.rewards.get(self.agent_selection, None)}")


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