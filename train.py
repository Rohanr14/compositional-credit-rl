"""
Main training script for Compositional Credit Assignment project.
Updated for Gym-MiniGrid and Instruction-Following support.
"""

import argparse
import os
import sys
import numpy as np
import torch
from tqdm import tqdm
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Use the new adapter instead of custom gridworld
from src.environments.minigrid_adapter import create_env
from src.agents.ppo_agent import PPOAgent
from src.agents.dqn_agent import DQNAgent
from src.agents.cca_agent import CCAAgent
from src.utils.metrics import MetricsTracker

def train(args):
    """Main training loop"""

    # 1. Create Environment (MiniGrid)
    # The adapter returns a dict observation: {'image': ..., 'mission': ...}
    env = create_env(args.task)

    print(f"\nEnvironment: {args.task} (MiniGrid)")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space.n}\n")

    # 2. Create Agent
    # We pass the full observation space so the agent can inspect shapes
    if args.agent == 'ppo':
        agent = PPOAgent(env.observation_space, env.action_space)
    elif args.agent == 'dqn':
        agent = DQNAgent(env.observation_space, env.action_space)
    elif args.agent == 'cca':
        agent = CCAAgent(env.observation_space, env.action_space)
    else:
        raise ValueError(f"Unknown agent: {args.agent}")

    print(f"Training agent: {args.agent.upper()}\n")

    metrics = MetricsTracker()
    training_data = {'rewards': [], 'lengths': [], 'success_rate': []}

    # 3. Training Loop
    pbar = tqdm(range(args.episodes), desc="Training")

    for episode in pbar:
        # Reset returns a DICT now
        obs_dict, _ = env.reset()

        # Extract components for the agent
        obs_img = obs_dict['image']       # Shape: (7, 7, 3)
        obs_mission = obs_dict['mission'] # Shape: (Seq_Len,)

        done = False
        truncated = False
        episode_reward = 0
        episode_length = 0

        # PPO Rollout Reset
        if args.agent == 'ppo':
            agent.reset_storage()

        while not done and not truncated:
            # === SELECT ACTION ===
            # All agents now accept image AND instruction (No blind baseline)
            if args.agent == 'cca':
                # CCA might use extra info for internal logging, but input is same
                task_info = {'task_composition': env.get_task_composition()}
                action, value = agent.select_action(obs_img, obs_mission, task_info=task_info)
            else:
                action, value = agent.select_action(obs_img, obs_mission)

            # === STEP ===
            next_obs_dict, reward, done, truncated, info = env.step(action)

            next_img = next_obs_dict['image']
            next_mission = next_obs_dict['mission']

            episode_reward += reward
            episode_length += 1

            # === STORE TRANSITION ===
            if args.agent == 'ppo':
                # PPO needs to store both inputs to replay them later
                agent.store_transition(
                    obs_img, obs_mission,
                    action, reward, value,
                    done or truncated
                )

            elif args.agent == 'dqn':
                agent.store_transition(
                    obs_img, obs_mission,
                    action, reward,
                    next_img, next_mission,
                    done or truncated
                )

            elif args.agent == 'cca':
                task_info = {'task_composition': env.get_task_composition()}
                agent.store_transition(
                    obs_img, obs_mission,
                    action, reward,
                    task_info
                )

            # Advance state
            obs_img = next_img
            obs_mission = next_mission

        # Episode complete
        success = (episode_reward > 0) # In Minigrid, any reward > 0 means success

        if args.agent == 'cca':
            agent.end_episode(success=success)

        # === UPDATES ===
        loss_info = {}

        if args.agent == 'ppo':
            if episode % args.update_freq == 0:
                loss_info = agent.update(next_img, next_mission)

        elif args.agent == 'dqn':
            if episode >= 100:
                loss_info = agent.update()

        elif args.agent == 'cca':
            # CCA updates from its internal successful memory
            if episode % args.update_freq == 0 and episode > 0:
                loss_info = agent.update(num_iterations=10)

        # Logging
        if loss_info:
            metrics.add_loss(loss_info)

        metrics.add_episode(episode_reward, episode_length, success)
        training_data['rewards'].append(episode_reward)
        training_data['lengths'].append(episode_length)
        training_data['success_rate'].append(1.0 if success else 0.0)

        if episode % 10 == 0:
            stats = metrics.get_stats(window=50)
            pbar.set_postfix({
                'R': f"{stats.get('mean_reward', 0):.2f}",
                'S': f"{stats.get('success_rate', 0):.2%}"
            })

        # Save Checkpoints
        if episode % args.save_freq == 0 and episode > 0:
            os.makedirs('checkpoints', exist_ok=True)
            agent.save(f"checkpoints/{args.agent}_{args.task}_{episode}.pt")

    # Final Save
    os.makedirs('checkpoints', exist_ok=True)
    agent.save(f"checkpoints/{args.agent}_{args.task}_final.pt")

    # Save Data
    os.makedirs('results', exist_ok=True)
    with open(f'results/{args.agent}_{args.task}_training.json', 'w') as f:
        json.dump(training_data, f)

    print("\nTraining complete!")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, default='cca', choices=['ppo', 'dqn', 'cca'])
    parser.add_argument('--task', type=str, default='move-pick-place')
    parser.add_argument('--episodes', type=int, default=5000)
    parser.add_argument('--update-freq', type=int, default=10)
    parser.add_argument('--save-freq', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train(args)