"""
Main trainingscript for Compositional Credit Assignment project.
Usage: python train.py - -agent cca - -task move - pick - place - -episodes 5000
"""

import argparse
import os
import sys
import numpy as np
import torch
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.environments.compositional_gridworld import create_task
from src.agents.ppo_agent import PPOAgent
from src.agents.dqn_agent import DQNAgent
from src.agents.cca_agent import CCAAgent
from src.utils.metrics import MetricsTracker
import json


def train(args):
    """Main training loop"""

    # Create environment
    env = create_task(args.task, grid_size=15)
    print(f"\\nEnvironment: {args.task}")
    print(f"Task composition: {env.task_composition}")
    print(f"Action space: {env.action_space.n}")
    print(f"Observation space: {env.observation_space.shape}\\n")

    # Create agent
    if args.agent == 'ppo':
        agent = PPOAgent(env.observation_space, env.action_space)
    elif args.agent == 'dqn':
        agent = DQNAgent(env.observation_space, env.action_space)
    elif args.agent == 'cca':
        agent = CCAAgent(env.observation_space, env.action_space)
    else:
        raise ValueError(f"Unknown agent: {args.agent}")

    print(f"Training agent: {args.agent.upper()}\\n")

    # Metrics tracker
    metrics = MetricsTracker()

    # Training results for saving
    training_data = {
        'rewards': [],
        'lengths': [],
        'success_rate': []
    }

    # Training loop
    pbar = tqdm(range(args.episodes), desc="Training")

    for episode in pbar:
        obs, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        episode_length = 0

        # For PPO: collect rollout
        if args.agent == 'ppo':
            agent.reset_storage()

        while not done and not truncated:
            # Select action
            if args.agent == 'cca':
                action, value = agent.select_action(obs, task_info=info)
            else:
                action, value = agent.select_action(obs)

            # Step environment
            next_obs, reward, done, truncated, info = env.step(action)

            episode_reward += reward
            episode_length += 1

            # Store transition
            if args.agent == 'ppo':
                with torch.no_grad():
                    obs_tensor = agent.preprocess_observation(obs)
                    action_logits, _ = agent.policy(obs_tensor)
                    log_probs = torch.nn.functional.log_softmax(action_logits, dim=-1)
                    action_log_prob = log_probs[0, action].item()

                agent.store_transition(obs, action, reward, value, action_log_prob, done or truncated)

            elif args.agent == 'dqn':
                agent.store_transition(obs, action, reward, next_obs, done or truncated)

            elif args.agent == 'cca':
                agent.store_transition(obs, action, reward, info)

            obs = next_obs

        # Episode complete
        success = episode_reward > 0.5  # Task completed successfully

        # End episode for CCA
        if args.agent == 'cca':
            agent.end_episode(success=success)

        # Update agent
        if args.agent == 'ppo':
            if episode % args.update_freq == 0:
                loss_info = agent.update(next_obs)
                if loss_info:
                    metrics.add_loss(loss_info)

        elif args.agent == 'dqn':
            if episode >= 100:  # Start updating after some experience
                for _ in range(10):
                    loss_info = agent.update()
                    if loss_info:
                        metrics.add_loss(loss_info)

        elif args.agent == 'cca':
            if episode % args.update_freq == 0 and episode > 0:
                loss_info = agent.update(num_iterations=10)
                if 'policy_loss' in loss_info:
                    metrics.add_loss(loss_info)

        # Track metrics
        metrics.add_episode(episode_reward, episode_length, success)
        training_data['rewards'].append(episode_reward)
        training_data['lengths'].append(episode_length)
        training_data['success_rate'].append(1.0 if success else 0.0)

        # Update progress bar
        if episode % 10 == 0:
            stats = metrics.get_stats(window=100)
            pbar.set_postfix({
                'reward': f"{stats.get('mean_reward', 0):.2f}",
                'success': f"{stats.get('success_rate', 0):.2%}",
                'length': f"{stats.get('mean_length', 0):.0f}"
            })

        # Print stats periodically
        if episode % args.print_freq == 0 and episode > 0:
            metrics.print_stats(window=100)

        # Save checkpoint
        if episode % args.save_freq == 0 and episode > 0:
            os.makedirs('checkpoints', exist_ok=True)
            save_path = f"checkpoints/{args.agent}_{args.task}_{episode}.pt"
            agent.save(save_path)
            print(f"Checkpoint saved: {save_path}")

    # Final save
    os.makedirs('checkpoints', exist_ok=True)
    final_path = f"checkpoints/{args.agent}_{args.task}_final.pt"
    agent.save(final_path)
    print(f"\\nFinal model saved: {final_path}")

    # Save training data
    os.makedirs('results', exist_ok=True)
    with open(f'results/{args.agent}_{args.task}_training.json', 'w') as f:
        json.dump(training_data, f)

    print("\\nTraining complete!")
    metrics.print_stats(window=100)

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL agents on compositional tasks")

    # Training args
    parser.add_argument('--agent', type=str, default='cca',
                        choices=['ppo', 'dqn', 'cca'],
                        help='Agent type')
    parser.add_argument('--task', type=str, default='move-pick-place',
                        choices=['move', 'pick', 'place', 'move-pick',
                                 'pick-place', 'move-pick-place',
                                 'move-avoid-pick', 'full-composition'],
                        help='Task composition')
    parser.add_argument('--episodes', type=int, default=5000,
                        help='Number of training episodes')
    parser.add_argument('--update-freq', type=int, default=10,
                        help='Update frequency (episodes)')
    parser.add_argument('--print-freq', type=int, default=100,
                        help='Print stats frequency (episodes)')
    parser.add_argument('--save-freq', type=int, default=1000,
                        help='Save checkpoint frequency (episodes)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train(args)