import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, List, Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from numpy import signedinteger
from numpy._typing import _64Bit


class CompositionalGridWorld(gym.Env):
    """A grid world environment with compositional task structure.

    Primitives:
    - MOVE: Navigate to a location
    - PICK: Collect an object
    - PLACE: Deposit an object
    - AVOID: Navigate while avoiding obstacles
    Tasks are compositions of primitives with sparse rewards."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    # Primitive action types
    PRIMITIVES = ["move", "pick", "place", "avoid"]

    # Object colors
    COLORS = ["red", "blue", "green"]

    def __init__(
            self,
            grid_size: int = 15,
            task_composition: List[str] = ["move", "pick", "place"],
            max_steps: int = 200,
            render_mode: Optional[str] = None
    ):
        super().__init__()

        self.grid_size = grid_size
        self.task_composition = task_composition
        self.max_steps = max_steps
        self.render_mode = render_mode

        # Action space: 4 directional movements + interact
        self.action_space = spaces.Discrete(5)  # UP, DOWN, LEFT, RIGHT, INTERACT

        # Observation space: grid + agent state + task info
        # Channels: [agent, red_obj, blue_obj, green_obj, goal, obstacles, carrying]
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(7, grid_size, grid_size),
            dtype=np.float32
        )

        # Initialize attributes that reset() will use
        self.agent_pos = None
        self.objects = {}
        self.goal_pos = None
        self.obstacles = []
        self.carrying = None
        self.completed_primitives = []
        self.grid = None
        self.current_step = 0
        self.task_progress = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.task_progress = 0  # Which primitive in the composition we're on

        # Initialize grid
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)

        # Place agent randomly
        self.agent_pos = self._random_free_pos()

        # Place objects based on task composition
        self.objects = {}
        for color_idx, color in enumerate(self.COLORS):
            self.objects[color] = self._random_free_pos()

        # Place goal location
        self.goal_pos = self._random_free_pos()

        # Place obstacles (for 'avoid' primitive)
        self.obstacles = []
        if "avoid" in self.task_composition:
            for _ in range(5):
                self.obstacles.append(self._random_free_pos())

        # Agent state
        self.carrying = None  # What object agent is carrying
        self.completed_primitives = []  # Track which primitives completed

        return self._get_obs(), self._get_info()

    def step(self, action):
        self.current_step += 1

        # Execute movement action
        if action < 4:  # Movement
            self._move(action)
        elif action == 4:  # Interact
            self._interact()

        # Check if current primitive is completed
        primitive_completed = self._check_primitive_completion()

        # Calculate reward (sparse - only on task completion)
        reward = 0.0
        terminated = False

        if primitive_completed:
            self.task_progress += 1
            # Small reward for completing a primitive
            reward = 0.1

            if self.task_progress >= len(self.task_composition):
                # Full task completed!
                reward = 1.0
                terminated = True

        # Check timeout
        truncated = self.current_step >= self.max_steps

        # Penalty for hitting obstacles (if avoid is in task)
        if "avoid" in self.task_composition:
            if self.agent_pos in self.obstacles:
                reward -= 0.5

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _move(self, action):
        """Execute movement action"""
        x, y = self.agent_pos

        if action == 0:  # UP
            x = max(0, x - 1)
        elif action == 1:  # DOWN
            x = min(self.grid_size - 1, x + 1)
        elif action == 2:  # LEFT
            y = max(0, y - 1)
        elif action == 3:  # RIGHT
            y = min(self.grid_size - 1, y + 1)

        self.agent_pos = (x, y)

    def _interact(self):
        """Handle interact action (pick/place)"""
        current_primitive = self.task_composition[self.task_progress]

        if current_primitive == "pick":
            # Check if agent is on an object
            for color, pos in self.objects.items():
                if self.agent_pos == pos and self.carrying is None:
                    self.carrying = color
                    self.objects[color] = None  # Remove from grid
                    break

        elif current_primitive == "place":
            # Check if agent is at goal and carrying something
            if self.agent_pos == self.goal_pos and self.carrying is not None:
                self.carrying = None  # Drop the object

    def _check_primitive_completion(self) -> bool:
        """Check if current primitive in composition is completed"""
        if self.task_progress >= len(self.task_composition):
            return False

        current_primitive = self.task_composition[self.task_progress]

        if current_primitive == "move":
            # Move to goal location
            target_color = self._get_target_color(self.task_progress)
            target_pos = self.objects.get(target_color)
            if target_pos and self.agent_pos == target_pos:
                return True

        elif current_primitive == "pick":
            # Successfully picked up target object
            target_color = self._get_target_color(self.task_progress)
            return self.carrying == target_color

        elif current_primitive == "place":
            # Successfully placed at goal
            return self.agent_pos == self.goal_pos and self.carrying is None

        elif current_primitive == "avoid":
            # Reached goal without hitting obstacles
            if self.agent_pos == self.goal_pos:
                return True

        return False

    def _get_target_color(self, primitive_idx: int) -> str:
        """Get target color for current primitive"""
        # Simple mapping: use color based on primitive index
        return self.COLORS[primitive_idx % len(self.COLORS)]

    def _random_free_pos(self) -> tuple[signedinteger[_64Bit], signedinteger[_64Bit]]:
        """Get random unoccupied position"""
        max_attempts = 1000
        for _ in range(max_attempts):
            pos = (
                self.np_random.integers(0, self.grid_size),
                self.np_random.integers(0, self.grid_size)
            )
            # Check if position is free
            is_agent = self.agent_pos is not None and pos == self.agent_pos
            is_object = pos in [v for v in self.objects.values() if v is not None]
            is_goal = self.goal_pos is not None and pos == self.goal_pos
            is_obstacle = pos in self.obstacles

            if not (is_agent or is_object or is_goal or is_obstacle):
                return pos

        # Fallback: return any position (shouldn't happen with 15x15 grid)
        return (
            self.np_random.integers(0, self.grid_size),
            self.np_random.integers(0, self.grid_size)
        )

    def _get_obs(self) -> np.ndarray:
        """Get observation tensor"""
        obs = np.zeros((7, self.grid_size, self.grid_size), dtype=np.float32)

        # Channel 0: Agent position
        obs[0, self.agent_pos[0], self.agent_pos[1]] = 1.0

        # Channels 1-3: Object positions (red, blue, green)
        for idx, color in enumerate(self.COLORS):
            if self.objects[color] is not None:
                pos = self.objects[color]
                obs[idx + 1, pos[0], pos[1]] = 1.0

        # Channel 4: Goal position
        obs[4, self.goal_pos[0], self.goal_pos[1]] = 1.0

        # Channel 5: Obstacles
        for obs_pos in self.obstacles:
            obs[5, obs_pos[0], obs_pos[1]] = 1.0

        # Channel 6: Carrying state (broadcast across grid)
        if self.carrying is not None:
            obs[6, :, :] = 1.0

        return obs

    def _get_info(self) -> Dict:
        """Get info dictionary"""
        return {
            "task_progress": self.task_progress,
            "completed_primitives": self.completed_primitives.copy(),
            "carrying": self.carrying,
            "task_composition": self.task_composition,
            "step": self.current_step
        }

    def render(self):
        if self.render_mode == "human":
            self._render_human()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()
        return None

    def _render_human(self):
        """Render environment for human viewing"""
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_aspect('equal')
        ax.grid(True)

        # Draw objects
        for color, pos in self.objects.items():
            if pos is not None:
                circle = Circle((pos[1] + 0.5, self.grid_size - pos[0] - 0.5),
                                0.3, color=color, alpha=0.7)
                ax.add_patch(circle)

        # Draw goal
        rect = Rectangle((self.goal_pos[1], self.grid_size - self.goal_pos[0] - 1),
                         1, 1, fill=False, edgecolor='gold', linewidth=3)
        ax.add_patch(rect)

        # Draw obstacles
        for obs_pos in self.obstacles:
            rect = Rectangle((obs_pos[1], self.grid_size - obs_pos[0] - 1),
                             1, 1, color='gray', alpha=0.5)
            ax.add_patch(rect)

        # Draw agent
        agent_marker = 'o' if self.carrying is None else 's'
        ax.plot(self.agent_pos[1] + 0.5, self.grid_size - self.agent_pos[0] - 0.5,
                agent_marker, markersize=15, color='black', markeredgewidth=2,
                markeredgecolor='white')

        # Add task info
        task_str = " → ".join(self.task_composition)
        progress_str = f"Progress: {self.task_progress}/{len(self.task_composition)}"
        ax.set_title(f"Task: {task_str}\\n{progress_str}", fontsize=12)

        plt.pause(0.1)
        plt.close()

    def _render_rgb_array(self):
        """Render as RGB array"""
        # Simple RGB rendering
        rgb = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)

        # White background
        rgb[:, :] = 255

        # Draw elements (simplified)
        rgb[self.agent_pos] = [0, 0, 0]  # Black agent

        return rgb


# Helper function to create standard task compositions
def create_task(task_name: str, grid_size: int = 15) -> CompositionalGridWorld:
    """Create environment with predefined task composition"""

    tasks = {
        "move": ["move"],
        "pick": ["pick"],
        "place": ["place"],
        "move-pick": ["move", "pick"],
        "pick-place": ["pick", "place"],
        "move-pick-place": ["move", "pick", "place"],
        "move-avoid-pick": ["move", "avoid", "pick"],
        "full-composition": ["move", "avoid", "pick", "place"],
    }

    if task_name not in tasks:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(tasks.keys())}")

    return CompositionalGridWorld(
        grid_size=grid_size,
        task_composition=tasks[task_name],
        max_steps=200
    )