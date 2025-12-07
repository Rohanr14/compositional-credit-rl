from abc import ABC, abstractmethod
import torch
import numpy as np


class BaseAgent(ABC):
    """Base class for all RL agents"""

    def __init__(self, observation_space, action_space, device='mps'):
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = torch.device(device if torch.backends.mps.is_available() else 'cpu')

        print(f"Using device: {self.device}")

    @abstractmethod
    def select_action(self, observation, deterministic=False):
        """Select action given observation"""
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        """Update agent parameters"""
        pass

    @abstractmethod
    def save(self, path: str):
        """Save agent"""
        pass

    @abstractmethod
    def load(self, path: str):
        """Load agent"""
        pass

    def preprocess_observation(self, obs):
        """Convert observation to tensor"""
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs).to(self.device)
        return obs.unsqueeze(0) if obs.dim() == 3 else obs