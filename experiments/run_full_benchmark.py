"""
Full benchmark suite for comparing all agents on all tasks.

This runs the complete experimental protocol for the paper:
    1. Train all agents on primitive tasks
    2. Train all agents on composite tasks
    3. Evaluate generalization to novel compositions
    4. Generate all plots and tables
Usage: python experiments / run_full_benchmark.py - -episodes 5000
"""

import argparse
import os
import sys
import subprocess
import json
from datetime import datetime
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.visualization import plot_training_curves, plot_generalization_results


def run_command(cmd):
    """Run shell command and capture output"""
    print(f"\\nRunning: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        return False
    return True


def main(args):
    """Run full benchmark"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    benchmark_dir = f"results/benchmark_{timestamp}"
    os.makedirs(benchmark_dir, exist_ok=True)

    print(f"\\n{'=' * 80}")
    print(f"RUNNING FULL BENCHMARK SUITE")
    print(f"Results will be saved to: {benchmark_dir}")
    print(f"{'=' * 80}\\n")

    agents = ['ppo', 'dqn', 'cca']

    # Tasks for comprehensive evaluation
    training_tasks = [
        'move-pick',  # Simple composition
        'pick-place',  # Simple composition
        'move-pick-place'  # Full composition
    ]

    all_results = {}

    # ========================================================================
    # PHASE 1: Train all agents on all tasks
    # ========================================================================
    print(f"\\n{'=' * 80}")
    print("PHASE 1: TRAINING")
    print(f"{'=' * 80}\\n")

    for task in training_tasks:
        print(f"\\n--- Training on task: {task} ---\\n")

        for agent in agents:
            print(f"Training {agent.upper()}...")

            cmd = f"python train.py --agent {agent} --task {task} --episodes {args.episodes} --seed {args.seed}"

            if not run_command(cmd):
                print(f"Failed to train {agent} on {task}")
                continue

            # Load training data
            try:
                with open(f'results/{agent}_{task}_training.json', 'r') as f:
                    training_data = json.load(f)

                if agent not in all_results:
                    all_results[agent] = {}
                all_results[agent][task] = training_data

            except Exception as e:
                print(f"Could not load training data: {e}")

    # ========================================================================
    # PHASE 2: Evaluate all agents
    # ========================================================================
    print(f"\\n{'=' * 80}")
    print("PHASE 2: EVALUATION")
    print(f"{'=' * 80}\\n")

    eval_results = {}

    for task in training_tasks:
        print(f"\\n--- Evaluating on task: {task} ---\\n")

        for agent in agents:
            print(f"Evaluating {agent.upper()}...")

            checkpoint = f"checkpoints/{agent}_{task}_final.pt"
            cmd = f"python evaluate.py --agent {agent} --task {task} --checkpoint {checkpoint} --num-episodes {args.eval_episodes}"

            if not run_command(cmd):
                print(f"Failed to evaluate {agent} on {task}")
                continue

            # Load eval data
            try:
                with open(f'results/{agent}_{task}_eval.json', 'r') as f:
                    eval_data = json.load(f)

                if agent not in eval_results:
                    eval_results[agent] = {}
                eval_results[agent][task] = eval_data

            except Exception as e:
                print(f"Could not load eval data: {e}")

    # ========================================================================
    # PHASE 3: Generate plots and tables
    # ========================================================================
    print(f"\\n{'=' * 80}")
    print("PHASE 3: GENERATING VISUALIZATIONS")
    print(f"{'=' * 80}\\n")

    # Plot training curves for main task
    main_task = 'move-pick-place'
    training_curves_data = {}

    for agent in agents:
        if agent in all_results and main_task in all_results[agent]:
            training_curves_data[agent] = all_results[agent][main_task]

    if training_curves_data:
        plot_training_curves(
            training_curves_data,
            save_path=f"{benchmark_dir}/training_curves.png"
        )

    # Plot generalization results
    gen_plot_data = {}

    for agent in agents:
        if agent in eval_results and main_task in eval_results[agent]:
            data = eval_results[agent][main_task]

            train_success = data['training_results']['success_rate']

            if data['generalization_results']:
                test_success = np.mean([
                    r['success_rate']
                    for r in data['generalization_results'].values()
                ])
            else:
                test_success = 0.0

            gen_plot_data[agent] = {
                'train_success': train_success,
                'test_success': test_success
            }

    if gen_plot_data:
        plot_generalization_results(
            gen_plot_data,
            save_path=f"{benchmark_dir}/generalization.png"
        )

    # ========================================================================
    # PHASE 4: Generate summary table
    # ========================================================================
    print(f"\\n{'=' * 80}")
    print("PHASE 4: SUMMARY")
    print(f"{'=' * 80}\\n")

    # Create summary table
    summary_table = []
    summary_table.append(f"{'Agent':<10} {'Task':<20} {'Train Success':<15} {'Test Success':<15} {'Gen Gap':<10}")
    summary_table.append("-" * 80)

    for agent in agents:
        for task in training_tasks:
            if agent in eval_results and task in eval_results[agent]:
                data = eval_results[agent][task]

                train_success = data['training_results']['success_rate']

                if data['generalization_results']:
                    test_success = np.mean([
                        r['success_rate']
                        for r in data['generalization_results'].values()
                    ])
                    gen_gap = train_success - test_success
                else:
                    test_success = 0.0
                    gen_gap = train_success

                summary_table.append(
                    f"{agent.upper():<10} {task:<20} {train_success:>14.2%} {test_success:>14.2%} {gen_gap:>9.2%}"
                )

    summary_text = "\\n".join(summary_table)
    print(summary_text)

    # Save summary
    with open(f"{benchmark_dir}/summary.txt", 'w') as f:
        f.write(summary_text)

    # Save all results
    with open(f"{benchmark_dir}/all_results.json", 'w') as f:
        json.dump({
            'training': all_results,
            'evaluation': eval_results
        }, f, indent=2)

    print(f"\\n{'=' * 80}")
    print(f"BENCHMARK COMPLETE!")
    print(f"Results saved to: {benchmark_dir}")
    print(f"{'=' * 80}\\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full benchmark suite")

    parser.add_argument('--episodes', type=int, default=5000,
                        help='Training episodes per agent/task')
    parser.add_argument('--eval-episodes', type=int, default=100,
                        help='Evaluation episodes')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    main(args)