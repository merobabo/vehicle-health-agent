from stable_baselines3 import PPO
from phase7_fleet_env import FleetMaintenanceEnv

env = FleetMaintenanceEnv(fleet_size=3)
model = PPO.load("fleet_rl_agent")

obs, _ = env.reset()

print("\n🧠 FLEET RL AGENT DECISIONS\n")

for step in range(5):
    actions, _ = model.predict(obs)

    obs, reward, _, _, _ = env.step(actions)

    print(f"Step {step+1}")
    print("Actions:", actions)
    print("Reward:", reward)
    print("-" * 50)
