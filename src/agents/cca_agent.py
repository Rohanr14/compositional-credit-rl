import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import List, Dict
from .base_agent import BaseAgent
from ..models.networks import ActorCritic
from ..models.credit_modules import CreditDecomposer, HindsightCreditAssignment


class CCAAgent(BaseAgent):
    """
    COMPOSITIONAL CREDIT ASSIGNMENT AGENT (Updated)

    Fixes:
    1. Accepts instruction tokens (avoids 'Blind Baseline').
    2. Corrects Advantage Math (Target - Value).
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

        # Core policy network (Updated to accept (image, instruction))
        # Note: input_channels is now 3 (RGB) for MiniGrid usually, or determined by wrapper
        self.policy = ActorCritic(
            input_shape=observation_space['image'].shape,
            action_dim=action_space.n,
            vocab_size=20  # Matches MinigridAdapter vocab
        ).to(self.device)

        # NOVEL COMPONENTS
        self.credit_decomposer = CreditDecomposer(
            feature_dim=128,
            num_primitives=4,
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
            'obs_images': [],
            'obs_missions': [],  # New: Store instructions
            'actions': [],
            'rewards': [],
            'features': [],
            'task_composition': []
        }

        self.successful_episodes = []

    def select_action(self, obs_img, obs_mission, task_info=None, deterministic=False):
        """
        Select action with compositional credit awareness.
        """
        # Preprocess
        img_tensor = torch.FloatTensor(obs_img).unsqueeze(0).to(self.device)
        mission_tensor = torch.LongTensor(obs_mission).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Get base policy action
            action_logits, value = self.policy(img_tensor, mission_tensor)

            # Extract features from the encoder for the credit modules
            features = self.policy.encoder(img_tensor, mission_tensor)

            # --- Credit Bias Logic (Optional: Only if you want transfer bias) ---
            if task_info and 'task_composition' in task_info:
                # Retrieve similar successful patterns from memory
                similar_patterns = self.credit_decomposer.retrieve_similar_patterns(
                    features, k=5
                )
                # Bias action distribution
                pattern_bias = self.policy.actor(similar_patterns) * 0.3
                action_logits = action_logits + pattern_bias
            # -------------------------------------------------------------------

            if deterministic:
                action = torch.argmax(action_logits, dim=-1)
            else:
                probs = F.softmax(action_logits, dim=-1)
                action = torch.multinomial(probs, 1).squeeze(-1)

        return action.cpu().item(), value.cpu().item()

    def store_transition(self, obs_img, obs_mission, action, reward, task_info):
        """Store transition components separately"""

        # Get features for storage
        img_tensor = torch.FloatTensor(obs_img).unsqueeze(0).to(self.device)
        mission_tensor = torch.LongTensor(obs_mission).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.policy.encoder(img_tensor, mission_tensor)

        self.current_episode['obs_images'].append(obs_img)
        self.current_episode['obs_missions'].append(obs_mission)
        self.current_episode['actions'].append(action)
        self.current_episode['rewards'].append(reward)
        self.current_episode['features'].append(features.squeeze(0))

        if task_info and 'task_composition' in task_info:
            self.current_episode['task_composition'] = task_info['task_composition']

    def end_episode(self, success=False):
        """Process completed episode"""
        if len(self.current_episode['obs_images']) == 0:
            return {}

        episode_data = self.current_episode.copy()

        if success:
            self.successful_episodes.append(episode_data)
            if len(self.successful_episodes) > 50:
                self.successful_episodes.pop(0)

        # Reset storage
        self.current_episode = {
            'obs_images': [],
            'obs_missions': [],
            'actions': [],
            'rewards': [],
            'features': [],
            'task_composition': []
        }

        return {'episode_stored': success}

    def update(self, num_iterations=10):
        """
        CCA Update with Corrected Math
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
                img_tensor = torch.FloatTensor(np.array(episode['obs_images'])).to(self.device)
                mission_tensor = torch.LongTensor(np.array(episode['obs_missions'])).to(self.device)
                features_tensor = torch.stack(episode['features'])
                actions_tensor = torch.LongTensor(episode['actions']).to(self.device)

                # Primitive sequence for credit decomposition
                primitive_sequence = self._task_to_primitives(episode['task_composition'])

                # === 1. HINDSIGHT CREDIT ASSIGNMENT ===
                # This gives us a score [0, 1] for every step
                final_reward = torch.FloatTensor([sum(episode['rewards'])]).to(self.device)
                hindsight_credits = self.hindsight_creditor(
                    features_tensor.unsqueeze(0),
                    final_reward
                ).squeeze(0)

                # === 2. COMPOSITIONAL DECOMPOSITION ===
                decomposed_credits, attention_weights = self.credit_decomposer(
                    features_tensor.unsqueeze(0),
                    primitive_sequence
                )
                decomposed_credits = decomposed_credits.squeeze(0)

                # Credit Loss: Make decomposition match hindsight
                credit_loss = F.mse_loss(decomposed_credits, hindsight_credits.detach())

                # === 3. POLICY UPDATE (CORRECTED MATH) ===
                # Re-evaluate actions
                action_logits, values = self.policy(img_tensor, mission_tensor)
                log_probs = F.log_softmax(action_logits, dim=-1)
                action_log_probs = log_probs.gather(1, actions_tensor.unsqueeze(-1)).squeeze(-1)

                # CRITICAL FIX: Advantage = Target - Baseline
                # We treat the hindsight credit as the "True Value" of that state
                target_value = hindsight_credits.detach()
                current_value = values.squeeze(-1)

                # Calculate Advantage
                advantage = target_value - current_value.detach()

                # Policy Loss (PPO-style or Vanilla PG)
                policy_loss = -(action_log_probs * advantage).mean()

                # Value Loss: Train critic to predict the hindsight credit
                value_loss = 0.5 * F.mse_loss(current_value, target_value)

                # Combined loss
                total_loss = policy_loss + value_loss + 0.5 * credit_loss

                # Update Policy
                self.policy_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.policy_optimizer.step()

                # Update Credit Modules
                self.credit_optimizer.zero_grad()
                credit_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.credit_decomposer.parameters()) +
                    list(self.hindsight_creditor.parameters()),
                    0.5
                )
                self.credit_optimizer.step()

                # Store pattern
                with torch.no_grad():
                    self.credit_decomposer.store_success_pattern(
                        features_tensor.unsqueeze(0),
                        primitive_sequence
                    )

                total_policy_loss += policy_loss.item()
                total_credit_loss += credit_loss.item()

        num_updates = num_iterations * len(episode_indices)

        return {
            'policy_loss': total_policy_loss / num_updates,
            'credit_loss': total_credit_loss / num_updates
        }

    def _task_to_primitives(self, task_composition: List[str]) -> List[int]:
        """Convert task composition to primitive indices"""
        # Updated for Minigrid mapping
        primitive_map = {'move': 0, 'pick': 1, 'place': 2, 'open': 3}
        return [primitive_map.get(prim, 0) for prim in task_composition]

    def save(self, path: str):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'credit_state_dict': self.credit_decomposer.state_dict(),
            'hindsight_state_dict': self.hindsight_creditor.state_dict(),
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.credit_decomposer.load_state_dict(checkpoint['credit_state_dict'])
        self.hindsight_creditor.load_state_dict(checkpoint['hindsight_state_dict'])