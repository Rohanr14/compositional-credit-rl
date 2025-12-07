import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List
from .base_agent import BaseAgent
from ..models.networks import ActorCritic
from ..models.credit_modules import CreditDecomposer, HindsightCreditAssignment


class CCAAgent(BaseAgent):
    """COMPOSITIONAL CREDIT ASSIGNMENT AGENT

    This is the novel contribution! Combines:
    1. Hindsight credit assignment on successful trajectories
    2. Compositional decomposition of credit across primitives
    3. Memory - augmented transfer to novel task compositions
    """

    def __init__(
            self,
            observation_space,
            action_space,
            lr=3e-4,
            gamma=0.99,
            device='mps'
    ):
        super().__init__(observation_space, action_space, device)

        self.gamma = gamma

        # Core policy network (same as PPO baseline for fair comparison)
        self.policy = ActorCritic(
            input_channels=observation_space.shape[0],
            action_dim=action_space.n,
            hidden_dim=128
        ).to(self.device)

        # NOVEL COMPONENTS
        self.credit_decomposer = CreditDecomposer(
            feature_dim=128,
            num_primitives=4,  # move, pick, place, avoid
            memory_size=100
        ).to(self.device)

        self.hindsight_creditor = HindsightCreditAssignment(
            feature_dim=128
        ).to(self.device)

        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.credit_optimizer = optim.Adam(
            list(self.credit_decomposer.parameters()) +
            list(self.hindsight_creditor.parameters()),
            lr=lr
        )

        # Storage
        self.current_episode = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'features': [],
            'task_composition': []
        }

        self.successful_episodes = []  # Store for hindsight learning

    def select_action(self, observation, task_info=None, deterministic=False):
        """
        Select action with compositional credit awareness.
        Uses stored credit patterns to bias exploration toward promising compositional strategies.
        """
        obs = self.preprocess_observation(observation)

        with torch.no_grad():
            # Get base policy action
            action_logits, value = self.policy(obs)
            features = self.policy.encoder(obs)

            # If we have task composition info, use credit guidance
            if task_info and 'task_composition' in task_info:
                primitive_sequence = self._task_to_primitives(task_info['task_composition'])

                # Retrieve similar successful patterns from memory
                similar_patterns = self.credit_decomposer.retrieve_similar_patterns(
                    features, k=5
                )

                # Bias action distribution toward successful patterns
                # (This is a key innovation: transfer from compositional memory)
                pattern_bias = torch.matmul(
                    similar_patterns.unsqueeze(1),
                    self.policy.actor[2].weight.t()
                ).squeeze(1) * 0.3  # Scaling factor

                action_logits = action_logits + pattern_bias

            # Sample action
            if deterministic:
                action = torch.argmax(action_logits, dim=-1)
            else:
                probs = torch.nn.functional.softmax(action_logits, dim=-1)
                action = torch.multinomial(probs, 1).squeeze(-1)

        return action.cpu().item(), value.cpu().item()

    def store_transition(self, obs, action, reward, task_info):
        """Store transition with task composition info"""
        obs_tensor = self.preprocess_observation(obs)

        with torch.no_grad():
            features = self.policy.encoder(obs_tensor)

        self.current_episode['observations'].append(obs)
        self.current_episode['actions'].append(action)
        self.current_episode['rewards'].append(reward)
        self.current_episode['features'].append(features.squeeze(0))

        if task_info and 'task_composition' in task_info:
            self.current_episode['task_composition'] = task_info['task_composition']

    def end_episode(self, success=False):
        """
        Process completed episode with hindsight credit assignment.
        This is where the magic happens! If episode was successful, we retroactively compute which states / actions deserved credit.
        """
        if len(self.current_episode['observations']) == 0:
            return {}

        episode_data = self.current_episode.copy()

        # Store successful episodes for later learning
        if success and episode_data['task_composition']:
            self.successful_episodes.append(episode_data)

            # Limit memory
            if len(self.successful_episodes) > 50:
                self.successful_episodes.pop(0)

        # Clear current episode
        self.current_episode = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'features': [],
            'task_composition': []
        }

        return {'episode_stored': success}

    def update(self, num_iterations=10):
        """
        CCA Update: Learn from successful episodes using compositional credit.
        Key innovations:
        1. Hindsight credit assignment on successful trajectories
        2. Compositional decomposition across primitives
        3.Store patterns in memory for transfer
        """
        if len(self.successful_episodes) < 5:
            return {'status': 'insufficient_data'}

        total_policy_loss = 0
        total_credit_loss = 0
        total_decomposition_accuracy = 0

        for _ in range(num_iterations):
            # Sample successful episodes
            batch_size = min(8, len(self.successful_episodes))
            episode_indices = np.random.choice(len(self.successful_episodes), batch_size, replace=False)

            for idx in episode_indices:
                episode = self.successful_episodes[idx]

                # Prepare trajectory data
                obs_tensor = torch.stack([
                    self.preprocess_observation(o).squeeze(0)
                    for o in episode['observations']
                ])
                features_tensor = torch.stack(episode['features'])
                actions_tensor = torch.LongTensor(episode['actions']).to(self.device)

                # Get task composition
                primitive_sequence = self._task_to_primitives(episode['task_composition'])

                # === HINDSIGHT CREDIT ASSIGNMENT ===
                final_reward = torch.FloatTensor([sum(episode['rewards'])]).to(self.device)
                hindsight_credits = self.hindsight_creditor(
                    features_tensor.unsqueeze(0),
                    final_reward
                ).squeeze(0)  # (seq_len,)

                # === COMPOSITIONAL CREDIT DECOMPOSITION ===
                decomposed_credits, attention_weights = self.credit_decomposer(
                    features_tensor.unsqueeze(0),
                    primitive_sequence
                )
                decomposed_credits = decomposed_credits.squeeze(0)  # (seq_len,)

                # Credit loss: hindsight and decomposed credits should agree
                credit_loss = nn.functional.mse_loss(decomposed_credits, hindsight_credits.detach())

                # === POLICY UPDATE WITH CREDIT-WEIGHTED ADVANTAGES ===
                # Re-evaluate actions with current policy
                action_logits, values = self.policy(obs_tensor)
                log_probs = torch.nn.functional.log_softmax(action_logits, dim=-1)
                action_log_probs = log_probs.gather(1, actions_tensor.unsqueeze(-1)).squeeze(-1)

                # Use hindsight credits as advantage estimates
                # (This is the key: credit assignment guides policy learning)
                advantages = hindsight_credits

                # Policy gradient loss weighted by credits
                policy_loss = -(action_log_probs * advantages.detach()).mean()

                # Value loss
                returns = self._compute_returns(episode['rewards'])
                returns_tensor = torch.FloatTensor(returns).to(self.device)
                value_loss = 0.5 * (returns_tensor - values.squeeze(-1)).pow(2).mean()

                # Combined loss
                total_loss = policy_loss + value_loss + 0.5 * credit_loss

                # Update policy
                self.policy_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.policy_optimizer.step()

                # Update credit modules separately
                self.credit_optimizer.zero_grad()
                credit_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.credit_decomposer.parameters()) +
                    list(self.hindsight_creditor.parameters()),
                    0.5
                )
                self.credit_optimizer.step()

                # === STORE SUCCESSFUL PATTERN IN COMPOSITIONAL MEMORY ===
                with torch.no_grad():
                    self.credit_decomposer.store_success_pattern(
                        features_tensor.unsqueeze(0),
                        primitive_sequence
                    )

                total_policy_loss += policy_loss.item()
                total_credit_loss += credit_loss.item()

                # Measure decomposition accuracy
                accuracy = (decomposed_credits - hindsight_credits).abs().mean()
                total_decomposition_accuracy += accuracy.item()

        num_updates = num_iterations * len(episode_indices)

        return {
            'policy_loss': total_policy_loss / num_updates,
            'credit_loss': total_credit_loss / num_updates,
            'decomposition_accuracy': total_decomposition_accuracy / num_updates,
            'memory_size': int(self.credit_decomposer.credit_memory_ptr.item())
        }

    def _task_to_primitives(self, task_composition: List[str]) -> List[int]:
        """Convert task composition to primitive indices"""
        primitive_map = {'move': 0, 'pick': 1, 'place': 2, 'avoid': 3}
        return [primitive_map.get(prim, 0) for prim in task_composition]

    def _compute_returns(self, rewards: List[float]) -> List[float]:
        """Compute discounted returns"""
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        return returns

    def save(self, path: str):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'credit_decomposer_state_dict': self.credit_decomposer.state_dict(),
            'hindsight_creditor_state_dict': self.hindsight_creditor.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'credit_optimizer_state_dict': self.credit_optimizer.state_dict(),
            'successful_episodes': self.successful_episodes
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.credit_decomposer.load_state_dict(checkpoint['credit_decomposer_state_dict'])
        self.hindsight_creditor.load_state_dict(checkpoint['hindsight_creditor_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.credit_optimizer.load_state_dict(checkpoint['credit_optimizer_state_dict'])
        if 'successful_episodes' in checkpoint:
            self.successful_episodes = checkpoint['successful_episodes']