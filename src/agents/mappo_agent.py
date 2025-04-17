import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from collections import deque


class OffloadingNetwork(nn.Module):
    """
    Neural network for the MAPPO agent to make offloading decisions
    """

    def __init__(self, input_dim, hidden_dim=128, lstm_dim=64):
        super(OffloadingNetwork, self).__init__()

        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # LSTM for capturing temporal dependencies
        self.lstm = nn.LSTMCell(hidden_dim, lstm_dim)

        # Policy head (Actor)
        self.policy_head = nn.Sequential(
            nn.Linear(lstm_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)  # 2 actions: local or offload
        )

        # Value head (Critic)
        self.value_head = nn.Sequential(
            nn.Linear(lstm_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Initialize weights
        self.apply(self._init_weights)

        # Internal state for LSTM
        self.lstm_hidden = None
        self.lstm_cell = None

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def reset_lstm_state(self, batch_size=1):
        """Reset LSTM hidden state and cell state."""
        device = next(self.parameters()).device
        self.lstm_hidden = torch.zeros(batch_size, self.lstm.hidden_size, device=device)
        self.lstm_cell = torch.zeros(batch_size, self.lstm.hidden_size, device=device)

    def forward(self, x):
        # Check shape and reshape if needed
        original_shape = x.shape
        if len(original_shape) > 2:
            # Reshape to (batch_size, features)
            x = x.view(original_shape[0], -1)

        batch_size = x.size(0)

        # Reset LSTM state if needed
        if self.lstm_hidden is None or self.lstm_cell is None or self.lstm_hidden.size(0) != batch_size:
            self.reset_lstm_state(batch_size)

        # Extract features
        features = self.feature_extractor(x)

        # LSTM update with features
        self.lstm_hidden, self.lstm_cell = self.lstm(
            features, (self.lstm_hidden, self.lstm_cell)
        )

        # Policy head (Actor)
        action_logits = self.policy_head(self.lstm_hidden)
        action_probs = F.softmax(action_logits, dim=-1).clamp(min=1e-6)

        # Value head (Critic)
        value = self.value_head(self.lstm_hidden)

        return action_probs, value


class MAPPOAgent:
    """
    Multi-Agent Proximal Policy Optimization agent for edge computing offloading
    """

    def __init__(self, observation_space, action_space, n_agents=3,
                 learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_param=0.2, value_loss_coef=0.5, entropy_coef=0.01,
                 max_grad_norm=0.5, hidden_dim=128, lstm_dim=64):
        """
        Initialize MAPPO agent

        Args:
            observation_space: Environment observation space
            action_space: Environment action space
            n_agents: Number of agents
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_param: PPO clipping parameter
            value_loss_coef: Value loss coefficient
            entropy_coef: Entropy coefficient
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.n_agents = n_agents
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        # Create networks and optimizers for each agent
        self.networks = []
        self.optimizers = []

        for _ in range(n_agents):
            # Determine input dimension from observation space
            if hasattr(observation_space, 'shape'):
                input_dim = observation_space.shape[0]
            else:
                input_dim = observation_space

            # Create network
            network = OffloadingNetwork(input_dim, hidden_dim, lstm_dim)
            optimizer = optim.Adam(network.parameters(), lr=learning_rate)

            self.networks.append(network)
            self.optimizers.append(optimizer)

        # Experience buffer
        self.reset_buffers()

        # Device to use
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move networks to device
        for i in range(n_agents):
            self.networks[i] = self.networks[i].to(self.device)

    def reset_buffers(self):
        """Reset experience buffers."""
        # For each agent, store: states, actions, log_probs, rewards, values, dones
        self.buffers = [{
            'states': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'values': [],
            'dones': []
        } for _ in range(self.n_agents)]

    def reset_lstm_states(self):
        """Reset LSTM states for all agents."""
        for i in range(self.n_agents):
            self.networks[i].reset_lstm_state()

    def act(self, states, agent_ids=None):
        """
        Select actions based on current states

        Args:
            states: List of states for each agent
            agent_ids: List of agent IDs to use (if None, use all agents)

        Returns:
            actions: List of actions
            log_probs: List of log probabilities
            values: List of value estimations
        """
        actions = []
        log_probs = []
        values = []

        # Default to all agents if agent_ids not specified
        if agent_ids is None:
            agent_ids = list(range(self.n_agents))

        for i, agent_id in enumerate(agent_ids):
            # Ensure state is torch tensor and valid
            if isinstance(states[i], list) or isinstance(states[i], np.ndarray):
                state = torch.FloatTensor(states[i]).unsqueeze(0).to(self.device)
            else:
                state = states[i].unsqueeze(0).to(self.device)

            # Check for NaN values
            if torch.isnan(state).any():
                print(f"Warning: NaN detected in state for agent {agent_id}")
                state = torch.nan_to_num(state, nan=0.0)

            # Forward pass
            with torch.no_grad():
                action_probs, value = self.networks[agent_id](state)

                # Check for NaN values in action_probs
                if torch.isnan(action_probs).any():
                    print(f"Warning: NaN detected in action_probs for agent {agent_id}")
                    action_probs = torch.ones_like(action_probs) / action_probs.size(-1)  # Uniform distribution

                # Sample action
                dist = Categorical(action_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            # Append results
            actions.append(action.item())
            log_probs.append(log_prob.item())
            values.append(value.item())

        return actions, log_probs, values

    def store_transition(self, agent_ids, states, actions, log_probs, rewards, values, dones):
        """
        Store transition in buffer

        Args:
            agent_ids: List of agent IDs
            states: List of states
            actions: List of actions
            log_probs: List of log probabilities
            rewards: List of rewards
            values: List of values
            dones: List of done flags
        """
        for i, agent_id in enumerate(agent_ids):
            self.buffers[agent_id]['states'].append(states[i])
            self.buffers[agent_id]['actions'].append(actions[i])
            self.buffers[agent_id]['log_probs'].append(log_probs[i])
            self.buffers[agent_id]['rewards'].append(rewards[i])
            self.buffers[agent_id]['values'].append(values[i])
            self.buffers[agent_id]['dones'].append(dones[i])

    def compute_gae(self, values, rewards, dones, next_value):
        """
        Compute Generalized Advantage Estimation (GAE)

        Args:
            values: List of values
            rewards: List of rewards
            dones: List of done flags
            next_value: Next value estimation

        Returns:
            advantages: GAE advantages
            returns: Returns for value loss
        """
        advantages = []
        returns = []
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            # Handle the case where dones[t] is a list
            done_val = int(any(dones[t])) if isinstance(dones[t], list) else int(dones[t])

            delta = rewards[t] + self.gamma * next_val * (1 - done_val) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - done_val) * gae

            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])

        return advantages, returns

    def update(self, next_values):
        """
        Update policy using PPO

        Args:
            next_values: Next value estimations for each agent

        Returns:
            update_info: Info on the update
        """
        update_info = []

        for i in range(self.n_agents):
            # Skip if buffer is empty
            if len(self.buffers[i]['states']) == 0:
                continue

            # Convert buffers to tensors - use numpy.array first to avoid the warning
            states_np = np.array(self.buffers[i]['states'])
            states = torch.FloatTensor(states_np).to(self.device)

            # Check for NaN values
            if torch.isnan(states).any():
                print(f"Warning: NaN detected in states for agent {i}")
                states = torch.nan_to_num(states, nan=0.0)

            actions = torch.LongTensor(self.buffers[i]['actions']).to(self.device)
            old_log_probs = torch.FloatTensor(self.buffers[i]['log_probs']).to(self.device)
            rewards = torch.FloatTensor(self.buffers[i]['rewards']).to(self.device)
            values = torch.FloatTensor(self.buffers[i]['values']).to(self.device)
            dones = self.buffers[i]['dones']

            # Compute GAE
            advantages, returns = self.compute_gae(values.cpu().numpy().tolist(),
                                                   rewards.cpu().numpy().tolist(),
                                                   dones,
                                                   next_values[i])

            # Convert to tensors
            advantages = torch.FloatTensor(advantages).to(self.device)
            returns = torch.FloatTensor(returns).to(self.device)

            # Check for NaN values in advantages and returns
            if torch.isnan(advantages).any():
                print(f"Warning: NaN detected in advantages for agent {i}")
                advantages = torch.zeros_like(advantages)

            if torch.isnan(returns).any():
                print(f"Warning: NaN detected in returns for agent {i}")
                returns = values.detach()  # Fall back to the original value estimates

            # Only normalize advantages if we have enough data
            if advantages.size(0) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Make sure all tensors have the right batch dimension
            batch_size = states.size(0)

            # Optimize PPO objective - do one update per batch rather than multiple passes
            self.optimizers[i].zero_grad()

            # Reset LSTM state before forward pass
            self.networks[i].reset_lstm_state(batch_size=batch_size)

            # Forward pass
            action_probs, value = self.networks[i](states)

            # Check for NaN values in action_probs and value
            if torch.isnan(action_probs).any():
                print(f"Warning: NaN detected in action_probs during update for agent {i}")
                self.reset_buffers()
                self.reset_lstm_states()
                continue  # Skip this update

            if torch.isnan(value).any():
                print(f"Warning: NaN detected in value during update for agent {i}")
                self.reset_buffers()
                self.reset_lstm_states()
                continue  # Skip this update

            try:
                # Compute the policy loss (surrogate objective)
                dist = Categorical(action_probs)
                new_log_probs = dist.log_prob(actions)

                # Make sure these tensors have compatible shapes
                ratio = torch.exp(new_log_probs - old_log_probs)

                # Reshape if needed - ensure both are of shape [batch_size]
                if ratio.shape != advantages.shape:
                    ratio = ratio.view(-1)
                    advantages = advantages.view(-1)

                # Clipped surrogate objective
                clip_adv = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages
                policy_loss = -torch.min(ratio * advantages, clip_adv).mean()

                # Fix the value shape issue - ensure both have the same shape
                value_squeezed = value.view(-1)
                returns_squeezed = returns.view(-1)

                # Compute value loss (MSE loss)
                value_loss = F.mse_loss(value_squeezed, returns_squeezed)

                # Compute entropy loss
                entropy_loss = dist.entropy().mean()

                # Total loss
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_loss

                # Skip update if loss is NaN
                if torch.isnan(loss).any():
                    print(f"Warning: NaN detected in loss for agent {i}")
                    self.reset_buffers()
                    self.reset_lstm_states()
                    continue  # Skip this update

                # Backprop and update
                loss.backward()
                nn.utils.clip_grad_norm_(self.networks[i].parameters(), self.max_grad_norm)
                self.optimizers[i].step()

                agent_info = {
                    "policy_loss": policy_loss.item(),
                    "value_loss": value_loss.item(),
                    "entropy_loss": entropy_loss.item()
                }
                update_info.append(agent_info)

            except Exception as e:
                print(f"Error during update for agent {i}: {e}")
                continue  # Skip this agent if there's an error

        # Reset buffers after update
        self.reset_buffers()

        # Reset LSTM states for all agents
        self.reset_lstm_states()

        return update_info