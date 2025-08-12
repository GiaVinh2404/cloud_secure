import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # Tăng số layer và số neuron, dùng LeakyReLU để tăng khả năng biểu diễn
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

class BaseDQNAgent:
    def __init__(
        self, 
        action_space, 
        obs_size, 
        lr=5e-4,                # Giảm learning rate để ổn định hơn
        gamma=0.995,            # Tăng gamma để agent chú ý phần thưởng dài hạn hơn
        epsilon=1.0, 
        epsilon_min=0.05,       # Cho phép agent tiếp tục khám phá nhiều hơn
        epsilon_decay=0.995, 
        device=None
    ):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_space = action_space
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.model = DQN(obs_size, action_space.n).to(self.device)
        self.target_model = DQN(obs_size, action_space.n).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = deque(maxlen=10000)  # Tăng kích thước replay buffer

    def act(self, observation):
        if np.random.rand() < self.epsilon:
            return self.action_space.sample()
        obs_t = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(obs_t)
        return int(torch.argmax(q_values).item())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size=64):  # Tăng batch size nếu đủ RAM
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Double DQN: dùng model để chọn action, target_model để tính Q-value
        with torch.no_grad():
            next_actions = self.model(next_states).argmax(1, keepdim=True)
            next_q_values = self.target_model(next_states).gather(1, next_actions).squeeze()
        q_values = self.model(states).gather(1, actions).squeeze()
        target_q = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, target_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save(self, filepath):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon