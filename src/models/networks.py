import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvEncoder(nn.Module):
    """Convolutional encoder for grid observations"""

    def __init__(self, input_channels: int = 7, hidden_dim: int = 128):
        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.fc = nn.Linear(64 * 15 * 15, hidden_dim)

    def forward(self, x):
        # x shape: (batch, 7, 15, 15)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc(x))

        return x


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO"""

    def __init__(self, input_channels: int = 7, action_dim: int = 5, hidden_dim: int = 128):
        super().__init__()

        self.encoder = ConvEncoder(input_channels, hidden_dim)

        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        features = self.encoder(x)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value

    def get_action(self, x, deterministic=False):
        """Sample action from policy"""
        action_logits, value = self.forward(x)

        if deterministic:
            action = torch.argmax(action_logits, dim=-1)
        else:
            probs = F.softmax(action_logits, dim=-1)
            action = torch.multinomial(probs, 1).squeeze(-1)

        return action, value

    def evaluate_actions(self, x, actions):
        """Evaluate actions for PPO update"""
        action_logits, value = self.forward(x)

        log_probs = F.log_softmax(action_logits, dim=-1)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)

        # Entropy for exploration
        probs = F.softmax(action_logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)

        return action_log_probs, value.squeeze(-1), entropy


class DQN(nn.Module):
    """Deep Q-Network"""

    def __init__(self, input_channels: int = 7, action_dim: int = 5, hidden_dim: int = 128):
        super().__init__()

        self.encoder = ConvEncoder(input_channels, hidden_dim)

        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        features = self.encoder(x)
        q_values = self.q_head(features)
        return q_values