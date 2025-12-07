import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .base_agent import BaseAgent
from ..models.networks import DQN
from ..utils.replay_buffer import ReplayBuffer


class DQNAgent(BaseAgent):
    """Deep Q-Network Agent (Baseline)"""

    def __init__(
            self,
            observation_space,
            action_space,
            lr=1e-3,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay=0.995,
            buffer_size=100000,
            batch_size=64,
            target_update_freq=1000,
            device='mps'
    ):
        super().__init__(observation_space, action_space, device)

        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Networks
        self.q_network = DQN(
            input_channels=observation_space.shape[0],
            action_dim=action_space.n
        ).to(self.device)

        self.target_network = DQN(
            input_channels=observation_space.shape[0],
            action_dim=action_space.n
        ).to(self.device)

        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        self.update_counter = 0

    def select_action(self, observation, deterministic=False):
        """Epsilon-greedy action selection"""
        if not deterministic and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_space.n), 0.0

        obs = self.preprocess_observation(observation)

        with torch.no_grad():
            q_values = self.q_network(obs)
            action = q_values.argmax(dim=-1)

        return action.cpu().item(), q_values.max().cpu().item()

    def store_transition(self, obs, action, reward, next_obs, done):
        """Store transition in replay buffer"""
        self.replay_buffer.add(obs, action, reward, next_obs, done)

    def update(self):
        """DQN update using replay buffer"""
        if len(self.replay_buffer) < self.batch_size:
            return {}

        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)
        obs, actions, rewards, next_obs, dones = batch

        # Convert to tensors
        obs = torch.FloatTensor(np.array(obs)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_obs = torch.FloatTensor(np.array(next_obs)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q values
        current_q = self.q_network(obs).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        # Target Q values
        with torch.no_grad():
            next_q = self.target_network(next_obs).max(dim=-1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # Loss
        loss = nn.functional.mse_loss(current_q, target_q)

        # Update
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()

        # Update target network
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return {
            'q_loss': loss.item(),
            'epsilon': self.epsilon,
            'avg_q': current_q.mean().item()
        }

    def save(self, path: str):
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']