import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class FleetMaintenanceEnv(gym.Env):
    """
    Centralized RL environment controlling a fleet of vehicles
    """

    def __init__(self, fleet_size=3):
        super(FleetMaintenanceEnv, self).__init__()

        self.fleet_size = fleet_size

        # Actions: 0=CONTINUE, 1=SCHEDULE, 2=IMMEDIATE (per vehicle)
        self.action_space = spaces.MultiDiscrete([3] * fleet_size)

        # State: [health, failure_prob] for each vehicle
        self.observation_space = spaces.Box(
            low=0,
            high=100,
            shape=(fleet_size * 2,),
            dtype=np.float32
        )

        self.vehicles = None

    def reset(self, seed=None, options=None):
        self.vehicles = []

        for _ in range(self.fleet_size):
            vehicle = {
                "health": random.randint(30, 100),
                "failure_prob": round(random.uniform(0.0, 1.0), 2),
                "failure_cost": random.choice([1000, 3000, 5000, 100000])
            }
            self.vehicles.append(vehicle)

        return self._get_state(), {}

    def _get_state(self):
        state = []
        for v in self.vehicles:
            state.extend([v["health"], v["failure_prob"]])
        return np.array(state, dtype=np.float32)

    def step(self, actions):
        reward = 0
        maintenance_count = 0
        terminated = False

        for v, action in zip(self.vehicles, actions):
            failure_happens = random.random() < v["failure_prob"]

            # Failure penalty
            if failure_happens:
                reward -= 100
            else:
                reward += 20

            # Maintenance costs
            if action == 2:  # IMMEDIATE
                reward -= 30
                maintenance_count += 1
            elif action == 1:  # SCHEDULE
                reward -= 10
                maintenance_count += 1

            # Risk-weighted cost
            reward -= v["failure_prob"] * v["failure_cost"]

            # Update vehicle state
            v["health"] = random.randint(30, 100)
            v["failure_prob"] = round(random.uniform(0.0, 1.0), 2)

        # Fleet congestion penalty
        if maintenance_count > 2:
            reward -= 50

        return self._get_state(), reward, terminated, False, {}
