import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict
import pandas as pd

sns.set_style("whitegrid")


def plot_training_curves(
        results: Dict[str, Dict],
        save_path: str = "results/training_curves.png"
):
    """Plot training curves comparing different agents"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Rewards over time
    ax = axes[0, 0]
    for agent_name, data in results.items():
        episodes = range(len(data['rewards']))
        # Smooth with moving average
        window = 50
        smoothed = pd.Series(data['rewards']).rolling(window=window, min_periods=1).mean()
        ax.plot(episodes, smoothed, label=agent_name, linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Reward')
    ax.set_title('Training Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Success rate over time
    ax = axes[0, 1]
    for agent_name, data in results.items():
        episodes = range(len(data['success_rate']))
        window = 50
        smoothed = pd.Series(data['success_rate']).rolling(window=window, min_periods=1).mean()
        ax.plot(episodes, smoothed, label=agent_name, linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate')
    ax.set_title('Task Success Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    # Plot 3: Episode lengths
    ax = axes[1, 0]
    for agent_name, data in results.items():
        episodes = range(len(data['lengths']))
        window = 50
        smoothed = pd.Series(data['lengths']).rolling(window=window, min_periods=1).mean()
        ax.plot(episodes, smoothed, label=agent_name, linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Length')
    ax.set_title('Episode Lengths')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Sample efficiency
    ax = axes[1, 1]
    agent_names = []
    samples_to_threshold = []

    threshold = 0.7  # 70% success rate
    for agent_name, data in results.items():
        success_rates = pd.Series(data['success_rate']).rolling(window=50, min_periods=1).mean()
        # Find first time we hit threshold
        indices = np.where(success_rates >= threshold)[0]
        if len(indices) > 0:
            samples = indices[0] * 200  # Approximate samples per episode
            agent_names.append(agent_name)
            samples_to_threshold.append(samples)

    if samples_to_threshold:
        bars = ax.bar(agent_names, samples_to_threshold)
        ax.set_ylabel('Environment Steps')
        ax.set_title(f'Sample Efficiency (Steps to {threshold:.0%} Success)')
        ax.grid(True, alpha=0.3, axis='y')

        # Color the bars
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        for bar, color in zip(bars, colors):
            bar.set_color(color)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")
    plt.close()


def plot_generalization_results(
        results: Dict[str, Dict],
        save_path: str = "results/generalization.png"
):
    """Plot generalization performance on novel task compositions"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Performance on training vs test tasks
    ax = axes[0]
    agent_names = list(results.keys())
    train_perf = [results[name]['train_success'] for name in agent_names]
    test_perf = [results[name]['test_success'] for name in agent_names]

    x = np.arange(len(agent_names))
    width = 0.35

    ax.bar(x - width / 2, train_perf, width, label='Training Tasks', alpha=0.8)
    ax.bar(x + width / 2, test_perf, width, label='Novel Compositions', alpha=0.8)

    ax.set_ylabel('Success Rate')
    ax.set_title('Generalization Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(agent_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.05])

    # Plot 2: Generalization gap
    ax = axes[1]
    gaps = [train_perf[i] - test_perf[i] for i in range(len(agent_names))]
    bars = ax.bar(agent_names, gaps)

    # Color based on gap size
    for i, bar in enumerate(bars):
        if gaps[i] < 0.2:
            bar.set_color('#45B7D1')  # Good generalization
        elif gaps[i] < 0.5:
            bar.set_color('#F7B731')  # Medium generalization
        else:
            bar.set_color('#FF6B6B')  # Poor generalization

    ax.set_ylabel('Generalization Gap')
    ax.set_title('Train-Test Performance Gap (Lower is Better)')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Generalization results saved to {save_path}")
    plt.close()


def visualize_credit_assignment(
        episode_data: Dict,
        save_path: str = "results/credit_visualization.png"
):
    """Visualize credit assignment for a single episode"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    steps = range(len(episode_data['rewards']))

    # Plot 1: Rewards over episode
    ax = axes[0]
    ax.plot(steps, episode_data['rewards'], 'o-', linewidth=2, markersize=4)
    ax.set_ylabel('Reward')
    ax.set_title('Episode Rewards')
    ax.grid(True, alpha=0.3)

    # Plot 2: Hindsight credits
    if 'hindsight_credits' in episode_data:
        ax = axes[1]
        ax.plot(steps, episode_data['hindsight_credits'], 'o-',
                color='#4ECDC4', linewidth=2, markersize=4)
        ax.set_ylabel('Hindsight Credit')
        ax.set_title('Hindsight Credit Assignment')
        ax.grid(True, alpha=0.3)

    # Plot 3: Compositional attention weights
    if 'attention_weights' in episode_data:
        ax = axes[2]
        attention = np.array(episode_data['attention_weights'])

        if attention.ndim == 2:
            im = ax.imshow(attention.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
            ax.set_ylabel('Primitive')
            ax.set_xlabel('Timestep')
            ax.set_title('Attention to Primitives Over Time')
            ax.set_yticks(range(attention.shape[1]))
            ax.set_yticklabels(['Move', 'Pick', 'Place', 'Avoid'][:attention.shape[1]])
            plt.colorbar(im, ax=ax, label='Attention Weight')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Credit visualization saved to {save_path}")
    plt.close()