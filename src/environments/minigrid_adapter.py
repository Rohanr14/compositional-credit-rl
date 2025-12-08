import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
import numpy as np


class MinigridAdapter(gym.ObservationWrapper):
    """
    Adapts Minigrid to your Compositional Agent.
    1. Standardizes 'move', 'pick', 'place' to specific Minigrid envs.
    2. Tokenizes the textual mission string for the agents.
    """

    TASK_MAPPING = {
        # Simple navigation
        'move': 'MiniGrid-Empty-8x8-v0',
        # Finding and picking up objects
        'pick': 'MiniGrid-Unlock-v0',
        # Complex composite tasks (Move -> Interact -> Move)
        'move-pick-place': 'MiniGrid-BlockedUnlockPickup-v0'
    }

    def __init__(self, task_name, max_steps=100):
        if task_name not in self.TASK_MAPPING:
            raise ValueError(f"Unknown task mapping for: {task_name}")

        env_id = self.TASK_MAPPING[task_name]
        env = gym.make(env_id, max_steps=max_steps, render_mode="rgb_array")
        # Use partial observability (standard for research)
        env = ImgObsWrapper(env)
        super().__init__(env)

        self.task_name = task_name

        # Define a simple vocabulary for instructions
        self.vocab = {"move": 1, "pick": 2, "place": 3, "open": 4, "ball": 5, "key": 6, "box": 7}
        self.max_seq_len = 10

        # Update observation space to include instruction
        self.observation_space = gym.spaces.Dict({
            'image': self.env.observation_space,  # 7x7x3 grid
            'mission': gym.spaces.Box(0, len(self.vocab), shape=(self.max_seq_len,), dtype=np.int32)
        })

    def observation(self, obs):
        """Standardize observation and tokenize mission"""
        # Tokenize the mission string (simple hash-based or predefined)
        # For this fix, we map your task structure to tokens
        mission_str = self.env.unwrapped.mission
        tokens = np.zeros(self.max_seq_len, dtype=np.int32)

        # Simple keyword extraction for the baseline to "see"
        words = mission_str.lower().split()
        for i, word in enumerate(words[:self.max_seq_len]):
            if word in self.vocab:
                tokens[i] = self.vocab[word]

        return {
            'image': obs,  # The grid image
            'mission': tokens  # The instruction vector
        }

    def get_task_composition(self):
        """Returns the logical composition for CCA (Ground Truth)"""
        if 'pick' in self.task_name:
            return ["move", "pick"]
        elif 'Blocked' in self.TASK_MAPPING[self.task_name]:
            return ["move", "pick", "move", "place"]
        return ["move"]


def create_env(task_name):
    return MinigridAdapter(task_name)