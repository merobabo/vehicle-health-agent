from stable_baselines3 import PPO
from phase7_rl_environment import VehicleMaintenanceEnv

env = VehicleMaintenanceEnv()
model = PPO.load("rl_maintenance_agent")

obs, _ = env.reset()

print("\n🧠 RL AGENT DECISIONS\n")

for step in range(5):
    action, _ = model.predict(obs)

    obs, reward, done, _, _ = env.step(action)

    print(f"Step {step+1}")
    print(f"State → Health={obs[0]}, Failure Risk={obs[1]}")
    print(f"Action → {['CONTINUE','SCHEDULE','IMMEDIATE'][action]}")
    print(f"Reward → {reward}")
    print("-" * 40)

    if done:
        obs, _ = env.reset()
