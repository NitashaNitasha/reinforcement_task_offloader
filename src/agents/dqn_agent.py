import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)


class DQNAgent:
    def __init__(self, observation_space, action_space,
                 learning_rate=1e-3, gamma=0.99,
                 epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.95,
                 buffer_size=10000, batch_size=1024, hidden_dim=128):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_dim = observation_space.shape[0]
        self.n_actions = action_space.n

        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        # Networks
        self.q_network = QNetwork(self.obs_dim, self.n_actions, hidden_dim).to(self.device)
        self.target_network = QNetwork(self.obs_dim, self.n_actions, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        # Experience replay
        self.memory = deque(maxlen=buffer_size)

        # Counter for updating target network
        self.update_freq = 100
        self.step_counter = 0


    def store_transition(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    def select_action(self, state):
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        else:
            state_tensor = torch.FloatTensor(np.array(state)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

    def train_step(self):
        """Sample from buffer and train Q-network."""
        if len(self.memory) < self.batch_size:
            return None
        # temporarily
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Compute target Q-values
        # Compute Q(s', a') for next state
        with torch.no_grad():
            next_q_values = self.q_network(next_states)
            max_next_q = next_q_values.max(dim=1, keepdim=True)[0]  # shape: (batch_size, 1)

        # Make sure reward and dones are shaped as (batch_size, 1)
        rewards = rewards.unsqueeze(1) if rewards.ndim == 1 else rewards
        dones = dones.unsqueeze(1) if dones.ndim == 1 else dones

        # Compute target Q-values
        target_q = rewards + self.gamma * max_next_q * (1 - dones)

        # Compute current Q-values
        current_q = self.q_network(states).gather(1, actions)

        loss = self.criterion(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.step_counter += 1
        if self.step_counter % self.update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        print(f"[DEBUG] Batch size: {len(self.memory)}")
        print(f"[DEBUG] Rewards mean: {rewards.mean().item()}")
        print(f"[DEBUG] Target Q mean: {target_q.mean().item()}")

        return loss.item()

    def save(self, filepath):
        torch.save(self.q_network.state_dict(), filepath)

    def load(self, filepath):
        self.q_network.load_state_dict(torch.load(filepath))
        self.target_network.load_state_dict(torch.load(filepath))
