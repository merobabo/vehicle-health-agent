import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class VehicleMaintenanceEnv(gym.Env):
    def __init__(self):
        super(VehicleMaintenanceEnv, self).__init__()

        # Actions: continue, schedule, immediate
        self.action_space = spaces.Discrete(3)

        # State: health (0–100), failure_prob (0–1)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0]),
            high=np.array([100.0, 1.0]),
            dtype=np.float32
        )

        self.state = None

    def reset(self, seed=None, options=None):
        health = random.randint(20, 100)
        failure_prob = round(random.uniform(0.0, 1.0), 2)
        self.state = np.array([health, failure_prob], dtype=np.float32)
        return self.state, {}

    def step(self, action):
        health, failure_prob = self.state

        reward = 0
        terminated = False

        # Failure simulation
        failure_occurs = random.random() < failure_prob

        if action == 0:  # CONTINUE
            if failure_occurs:
                reward = -100
                terminated = True
            else:
                reward = +10

        elif action == 1:  # SCHEDULE
            reward = +30
            terminated = True

        elif action == 2:  # IMMEDIATE
            reward = +50
            terminated = True

        # New random state
        self.state = np.array([
            random.randint(20, 100),
            round(random.uniform(0.0, 1.0), 2)
        ], dtype=np.float32)

        return self.state, reward, terminated, False, {}
