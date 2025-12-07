import numpy as np
from typing import Dict


class MetricsTracker:
    """Track and compute training metrics"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rate = []
        self.losses = []

    def add_episode(self, reward: float, length: int, success: bool):
        """Add episode statistics"""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.success_rate.append(1.0 if success else 0.0)

    def add_loss(self, loss: Dict):
        """Add loss values"""
        self.losses.append(loss)

    def get_stats(self, window: int = 100) -> Dict:
        """Get statistics over recent window"""
        if len(self.episode_rewards) == 0:
            return {}

        recent_rewards = self.episode_rewards[-window:]
        recent_lengths = self.episode_lengths[-window:]
        recent_success = self.success_rate[-window:]

        stats = {
            'mean_reward': np.mean(recent_rewards),
            'std_reward': np.std(recent_rewards),
            'mean_length': np.mean(recent_lengths),
            'success_rate': np.mean(recent_success),
            'total_episodes': len(self.episode_rewards)
        }

        if self.losses:
            recent_losses = self.losses[-window:]
            for key in recent_losses[0].keys():
                values = [loss[key] for loss in recent_losses if key in loss]
                if values:
                    stats[f'mean_{key}'] = np.mean(values)

        return stats

    def print_stats(self, window: int = 100):
        """Print recent statistics"""
        stats = self.get_stats(window)

        print(f"\\n{'=' * 60}")
        print(f"Episodes: {stats.get('total_episodes', 0)}")
        print(f"Mean Reward: {stats.get('mean_reward', 0):.2f} ± {stats.get('std_reward', 0):.2f}")
        print(f"Mean Length: {stats.get('mean_length', 0):.1f}")
        print(f"Success Rate: {stats.get('success_rate', 0):.2%}")

        for key, value in stats.items():
            if key.startswith('mean_') and key not in ['mean_reward', 'mean_length']:
                print(f"{key}: {value:.4f}")

        print(f"{'=' * 60}\\n")