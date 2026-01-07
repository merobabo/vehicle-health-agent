from stable_baselines3 import PPO
from phase7_fleet_env import FleetMaintenanceEnv

env = FleetMaintenanceEnv(fleet_size=3)

model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1
)

print("🚀 Training Fleet RL Agent...")
model.learn(total_timesteps=20000)

model.save("fleet_rl_agent")
print("✅ Fleet RL Agent Trained & Saved")
