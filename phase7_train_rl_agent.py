from stable_baselines3 import PPO
from phase7_rl_environment import VehicleMaintenanceEnv

env = VehicleMaintenanceEnv()

model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1
)

print("🚀 Training RL Agent...")
model.learn(total_timesteps=10000)

model.save("rl_maintenance_agent")
print("✅ RL Agent Trained & Saved")
