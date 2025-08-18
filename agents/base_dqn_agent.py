import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import List, Tuple, Deque, Any, Dict
from config.settings import DQN_CONFIG


# =========================
#   Neural Network for DQN
# =========================
class DQN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_layers: List[int] = None):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [256, 128, 64]

        layers: List[nn.Module] = []
        last_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        layers.append(nn.Linear(last_dim, output_dim))
        self.fc = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# =========================
#        Base DQN Agent
# =========================
class BaseDQNAgent:
    """
    DQN agent (Double DQN) với:
      - Epsilon-greedy policy
      - Replay buffer
      - Target network
      - Epsilon decay THEO EPISODE (gọi agent.decay_epsilon() sau mỗi episode)
    """

    def __init__(self, action_space, obs_size: int, config: Dict[str, Any] = DQN_CONFIG, device: torch.device = None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_space = action_space

        # Hyperparameters
        self.gamma: float = float(config.get("gamma", 0.99))
        self.epsilon: float = float(config.get("epsilon_start", 1.0))
        self.epsilon_min: float = float(config.get("epsilon_min", 0.05))
        self.epsilon_decay: float = float(config.get("epsilon_decay", 0.995))
        self.batch_size: int = int(config.get("batch_size", 128))
        self.target_update: int = int(config.get("target_update", 500))
        self.lr: float = float(config.get("lr", 5e-5))
        self.hidden_layers: List[int] = list(config.get("hidden_layers", [256, 128, 64]))

        # Networks
        n_actions = getattr(action_space, "n", None)
        if n_actions is None:
            raise ValueError("action_space phải có thuộc tính .n (Discrete).")

        self.model = DQN(obs_size, n_actions, self.hidden_layers).to(self.device)
        self.target_model = DQN(obs_size, n_actions, self.hidden_layers).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        # Optimizer & loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.SmoothL1Loss()  # Huber loss ổn định hơn MSE

        # Replay buffer
        self.memory: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(
            maxlen=int(config.get("memory_size", 50000))
        )

        # Counters
        self.step_count: int = 0

        # Optional: gradient clipping
        self.grad_clip_max_norm: float = float(config.get("grad_clip_max_norm", 10.0))

    # ------------- Acting -------------

    def act(self, observation: np.ndarray) -> int:
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return self.action_space.sample()

        obs_t = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(obs_t)
            action = int(q_values.argmax(dim=1).item())
        return action

    # ------------- Memory -------------

    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.memory.append((state, action, reward, next_state, done))

    # ------------- Learning -------------

    def replay(self) -> None:
        """Một bước train từ replay buffer. KHÔNG giảm epsilon ở đây (decay theo episode)."""
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.as_tensor(np.array(states, dtype=np.float32), device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(np.array(next_states, dtype=np.float32), device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        # ---- Double DQN target ----
        with torch.no_grad():
            # Online net chọn action tốt nhất tại next_state
            next_actions = self.model(next_states_t).argmax(dim=1, keepdim=True)
            # Target net đánh giá Q cho action đó
            next_q_values = self.target_model(next_states_t).gather(1, next_actions).squeeze(1)
            target_q = rewards_t + self.gamma * next_q_values * (1.0 - dones_t)

        # Q(s,a) hiện tại
        current_q = self.model(states_t).gather(1, actions_t).squeeze(1)

        loss = self.criterion(current_q, target_q)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient clipping
        if self.grad_clip_max_norm is not None and self.grad_clip_max_norm > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_max_norm)

        self.optimizer.step()

        # Cập nhật target network theo chu kỳ cứng
        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.update_target()

    def update_target(self) -> None:
        """Hard update: θ_target ← θ_online."""
        self.target_model.load_state_dict(self.model.state_dict())

    # ------------- Exploration schedule -------------

    def decay_epsilon(self) -> None:
        """Giảm epsilon THEO EPISODE. Gọi hàm này sau khi kết thúc mỗi episode."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min

    # ------------- Checkpoint I/O -------------

    def get_state(self) -> Dict[str, Any]:
        """Lưu toàn bộ state cần thiết để resume training mượt mà."""
        return {
            "model_state_dict": self.model.state_dict(),
            "target_model_state_dict": self.target_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "step_count": self.step_count,
            "config": {
                "gamma": self.gamma,
                "epsilon_min": self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
                "batch_size": self.batch_size,
                "target_update": self.target_update,
                "lr": self.lr,
                "hidden_layers": self.hidden_layers,
            },
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Tải checkpoint. Chịu được cả state đầy đủ/thiếu.
        Ưu tiên các key chuẩn; nếu thiếu thì load phần có thể.
        """
        try:
            # Cho phép load từ 2 dạng: full dict hoặc đã bóc sẵn các phần
            model_sd = state_dict.get("model_state_dict", state_dict)
            target_sd = state_dict.get("target_model_state_dict", state_dict)
            opt_sd = state_dict.get("optimizer_state_dict", None)

            self.model.load_state_dict(model_sd, strict=False)
            self.target_model.load_state_dict(target_sd, strict=False)

            if opt_sd is not None:
                self.optimizer.load_state_dict(opt_sd)

            # Restore epsilon & counters nếu có
            self.epsilon = float(state_dict.get("epsilon", self.epsilon))
            self.step_count = int(state_dict.get("step_count", self.step_count))

            print("[INFO] Checkpoint loaded successfully.")
        except Exception as e:
            print(f"[WARN] Could not load full checkpoint: {e}. Attempting partial load...")
            try:
                # Thử load tối thiểu model
                if "model_state_dict" in state_dict:
                    self.model.load_state_dict(state_dict["model_state_dict"], strict=False)
                if "target_model_state_dict" in state_dict:
                    self.target_model.load_state_dict(state_dict["target_model_state_dict"], strict=False)
                self.epsilon = float(state_dict.get("epsilon", self.epsilon))
                self.step_count = int(state_dict.get("step_count", self.step_count))
                print("[INFO] Partial checkpoint loaded.")
            except Exception as e2:
                print(f"[ERROR] Failed to load even partial checkpoint: {e2}")
