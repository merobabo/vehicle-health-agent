import random
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from vehicle_adapter import get_vehicle_config

print("✅ Script started")

# ------------------ HEALTH SCORE ------------------
def calculate_health(temp, rpm, vibration, config):
    score = 100

    if temp > config["temp_limit"]:
        score -= 30

    if config["rpm_limit"] is not None and rpm > config["rpm_limit"]:
        score -= 20

    if vibration > 0.05:
        score -= 40

    return max(score, 0)

# ------------------ COST-AWARE AGENT ------------------
def intelligent_agent(health, failure_prob, failure_cost):
    expected_loss = failure_prob * failure_cost

    if expected_loss > 20000:
        return "IMMEDIATE_MAINTENANCE"
    elif expected_loss > 5000 or health < 50:
        return "SCHEDULE_MAINTENANCE"
    else:
        return "CONTINUE_OPERATION"

# ------------------ LOAD & TRAIN MODEL ------------------
print("✅ Loading dataset")

data = pd.read_csv("data/vehicle_data.csv")

X = data[["engine_temp", "rpm", "vibration"]]
y = data["failed"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

print("✅ Model trained")

# ------------------ VEHICLE LIST ------------------
vehicles = [
    {"type": "car"},
    {"type": "truck"},
    {"type": "aircraft"},
    {"type": "bike"},
    {"type": "scooter"}
]

print("\n🚀 Multi-Vehicle Intelligent Agent Running\n")
print("Vehicle list:", vehicles)
print("-" * 60)

# ------------------ SIMULATION LOOP ------------------
for v in vehicles:
    print(f"\n➡️ Processing vehicle: {v['type']}")

    config = get_vehicle_config(v["type"])

    # -------- Vehicle-specific sensor simulation --------
    if v["type"] == "car":
        temp = random.randint(80, 110)
        rpm = random.randint(2000, 4000)

    elif v["type"] == "truck":
        temp = random.randint(90, 130)
        rpm = random.randint(1800, 3200)

    elif v["type"] == "aircraft":
        temp = random.randint(600, 950)
        rpm = 0

    elif v["type"] == "bike":
        temp = random.randint(85, 115)
        rpm = random.randint(4000, 9000)

    elif v["type"] == "scooter":
        temp = random.randint(80, 105)
        rpm = random.randint(3000, 7500)

    vibration = round(random.uniform(0.02, 0.10), 2)

    health = calculate_health(temp, rpm, vibration, config)

    failure_prob = model.predict_proba(
        [[temp, rpm, vibration]]
    )[0][1]

    decision = intelligent_agent(
        health,
        failure_prob,
        config["failure_cost"]
    )

    print(f"Vehicle Type → {v['type'].upper()}")
    print(f"Temp={temp}, RPM={rpm}, Vibration={vibration}")
    print(f"Health={health}, Failure Risk={round(failure_prob, 2)}")
    print(f"Decision → {decision}")
    print("-" * 60)

print("\n✅ Script finished successfully")
