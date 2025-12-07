import torch
import torch.optim as optim
import numpy as np
from .base_agent import BaseAgent
from ..models.networks import ActorCritic


class PPOAgent(BaseAgent):
    """Proximal Policy Optimization Agent (Baseline)"""

    def __init__(
            self,
            observation_space,
            action_space,
            lr=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            value_coef=0.5,
            entropy_coef=0.01,
            device='mps'
    ):
        super().__init__(observation_space, action_space, device)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        # Networks
        self.policy = ActorCritic(
            input_channels=observation_space.shape[0],
            action_dim=action_space.n
        ).to(self.device)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Storage for rollouts
        self.reset_storage()

    def reset_storage(self):
        """Reset rollout storage"""
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def select_action(self, observation, deterministic=False):
        """Select action using current policy"""
        obs = self.preprocess_observation(observation)

        with torch.no_grad():
            action, value = self.policy.get_action(obs, deterministic)

        return action.cpu().item(), value.cpu().item()

    def store_transition(self, obs, action, reward, value, log_prob, done):
        """Store transition in rollout buffer"""
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def compute_gae(self, next_value):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        gae = 0

        values = self.values + [next_value]

        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + self.gamma * values[t + 1] * (1 - self.dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, self.values)]

        return advantages, returns

    def update(self, next_obs, num_epochs=4, batch_size=64):
        """PPO update"""
        if len(self.observations) == 0:
            return {}

        # Get final value for GAE
        next_obs_tensor = self.preprocess_observation(next_obs)
        with torch.no_grad():
            _, next_value = self.policy(next_obs_tensor)
            next_value = next_value.cpu().item()

        # Compute advantages and returns
        advantages, returns = self.compute_gae(next_value)

        # Convert to tensors
        obs_tensor = torch.stack([self.preprocess_observation(o).squeeze(0) for o in self.observations])
        actions_tensor = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)

        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        # PPO epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0

        dataset_size = len(self.observations)

        for _ in range(num_epochs):
            # Mini-batch updates
            indices = np.random.permutation(dataset_size)

            for start in range(0, dataset_size, batch_size):
                end = min(start + batch_size, dataset_size)
                batch_indices = indices[start:end]

                # Get batch
                batch_obs = obs_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]

                # Evaluate actions
                log_probs, values, entropy = self.policy.evaluate_actions(batch_obs, batch_actions)

                # Policy loss (PPO clipped)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = 0.5 * (batch_returns - values).pow(2).mean()

                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()

                # Update
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()

        # Clear storage
        self.reset_storage()

        num_updates = num_epochs * (dataset_size // batch_size + 1)

        return {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates
        }

    def save(self, path: str):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])