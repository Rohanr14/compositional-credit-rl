"""
Evaluation script for trained agents.

Tests:
1. Performance on training tasks
2. Generalization to novel task compositions
3. Sample efficiency analysis
Usage: python evaluate.py - -agent cca - -task move - pick - place - -checkpoint checkpoints / cca_final.pt
"""

import argparse
import os
import sys
import numpy as np
import torch
from tqdm import tqdm
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from environments.compositional_gridworld import create_task
from agents.ppo_agent import PPOAgent
from agents.dqn_agent import DQNAgent
from agents.cca_agent import CCAAgent
from utils.visualization import plot_generalization_results


def evaluate_agent(agent, env, num_episodes=100, deterministic=True):
    """Evaluate agent on environment"""

    successes = []
    rewards = []
    lengths = []

    for _ in tqdm(range(num_episodes), desc="Evaluating"):
        obs, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        episode_length = 0

        while not done and not truncated:
            if hasattr(agent, 'select_action'):
                if isinstance(agent, type(agent)) and agent.__class__.__name__ == 'CCAAgent':
                    action, _ = agent.select_action(obs, task_info=info, deterministic=deterministic)
                else:
                    action, _ = agent.select_action(obs, deterministic=deterministic)
            else:
                action = env.action_space.sample()

            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1

        successes.append(episode_reward > 0.5)
        rewards.append(episode_reward)
        lengths.append(episode_length)

    return {
        'success_rate': np.mean(successes),
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'mean_length': np.mean(lengths)
    }


def test_generalization(agent, training_task, num_episodes=100):
    """Test generalization to novel task compositions"""

    # Define novel test tasks (not seen during training)
    novel_tasks = {
        'move': ['pick-place', 'move-pick-place'],
        'pick': ['move-pick', 'move-pick-place'],
        'place': ['pick-place', 'move-pick-place'],
        'move-pick': ['move-pick-place', 'move-avoid-pick'],
        'pick-place': ['move-pick-place'],
        'move-pick-place': ['full-composition'],
        'move-avoid-pick': ['full-composition'],
    }

    test_tasks = novel_tasks.get(training_task, [])
    if not test_tasks:
        print(f"No novel compositions defined for training task: {training_task}")
        return {}

    results = {}

    for task_name in test_tasks:
        print(f"\\nTesting on novel composition: {task_name}")
        env = create_task(task_name)

        task_results = evaluate_agent(agent, env, num_episodes=num_episodes)
        results[task_name] = task_results

        print(f"  Success Rate: {task_results['success_rate']:.2%}")
        print(f"  Mean Reward: {task_results['mean_reward']:.3f} ± {task_results['std_reward']:.3f}")

        env.close()

    return results


def main(args):
    """Main evaluation"""

    print(f"\\n{'=' * 60}")
    print(f"EVALUATING {args.agent.upper()} on {args.task}")
    print(f"{'=' * 60}\\n")

    # Create environment
    env = create_task(args.task)

    # Load agent
    if args.agent == 'ppo':
        agent = PPOAgent(env.observation_space, env.action_space)
    elif args.agent == 'dqn':
        agent = DQNAgent(env.observation_space, env.action_space)
    elif args.agent == 'cca':
        agent = CCAAgent(env.observation_space, env.action_space)
    else:
        raise ValueError(f"Unknown agent: {args.agent}")

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        agent.load(args.checkpoint)
    else:
        print("WARNING: No checkpoint loaded, using randomly initialized agent")

    # Evaluate on training task
    print(f"\\n1. Performance on Training Task: {args.task}")
    print("-" * 60)
    train_results = evaluate_agent(agent, env, num_episodes=args.num_episodes)

    print(f"\\nTraining Task Results:")
    print(f"  Success Rate: {train_results['success_rate']:.2%}")
    print(f"  Mean Reward: {train_results['mean_reward']:.3f} ± {train_results['std_reward']:.3f}")
    print(f"  Mean Length: {train_results['mean_length']:.1f}")

    # Test generalization
    print(f"\\n2. Generalization to Novel Compositions")
    print("-" * 60)
    gen_results = test_generalization(agent, args.task, num_episodes=args.num_episodes)

    if gen_results:
        avg_gen_success = np.mean([r['success_rate'] for r in gen_results.values()])
        gen_gap = train_results['success_rate'] - avg_gen_success

        print(f"\\nGeneralization Summary:")
        print(f"  Training Success: {train_results['success_rate']:.2%}")
        print(f"  Test Success: {avg_gen_success:.2%}")
        print(f"  Generalization Gap: {gen_gap:.2%}")

    # Save results
    os.makedirs('results', exist_ok=True)
    all_results = {
        'agent': args.agent,
        'task': args.task,
        'training_results': train_results,
        'generalization_results': gen_results
    }

    with open(f'results/{args.agent}_{args.task}_eval.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\\nResults saved to results/{args.agent}_{args.task}_eval.json")

    env.close()

    print(f"\\n{'=' * 60}")
    print("EVALUATION COMPLETE")
    print(f"{'=' * 60}\\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained RL agents")

    parser.add_argument('--agent', type=str, required=True,
                        choices=['ppo', 'dqn', 'cca'],
                        help='Agent type')
    parser.add_argument('--task', type=str, required=True,
                        help='Task to evaluate on')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint file')
    parser.add_argument('--num-episodes', type=int, default=100,
                        help='Number of evaluation episodes')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    main(args)