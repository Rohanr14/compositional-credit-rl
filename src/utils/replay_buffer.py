from collections import deque
import random


class ReplayBuffer:
    """Experience replay buffer for off-policy algorithms"""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def add(self, obs, action, reward, next_obs, done):
        """Add experience to buffer"""
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int):
        """Sample random batch"""
        batch = random.sample(self.buffer, batch_size)

        obs, actions, rewards, next_obs, dones = zip(*batch)

        return obs, actions, rewards, next_obs, dones

    def __len__(self):
        return len(self.buffer)