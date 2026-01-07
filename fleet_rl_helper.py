import numpy as np
from stable_baselines3 import PPO
from phase7_fleet_env import FleetMaintenanceEnv

# Load trained fleet RL agent
fleet_env = FleetMaintenanceEnv(fleet_size=3)
fleet_agent = PPO.load("fleet_rl_agent")

ACTION_MAP = {
    0: "✅ CONTINUE",
    1: "⚠️ SCHEDULE MAINTENANCE",
    2: "🚨 IMMEDIATE MAINTENANCE"
}

def fleet_rl_decision(vehicles_state):
    """
    vehicles_state = [
        (health1, failure_prob1),
        (health2, failure_prob2),
        (health3, failure_prob3)
    ]
    """

    # Flatten state for RL model
    obs = []
    for h, p in vehicles_state:
        obs.extend([h, p])

    obs = np.array(obs, dtype=np.float32)

    actions, _ = fleet_agent.predict(obs)

    return [ACTION_MAP[int(a)] for a in actions]
