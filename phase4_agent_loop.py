import pandas as pd
import time
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ------------------ HEALTH SCORE ------------------
def calculate_health(temp, rpm, vibration):
    score = 100
    if temp > 95:
        score -= 20
    if rpm > 3500:
        score -= 20
    if vibration > 0.05:
        score -= 40
    return max(score, 0)

# ------------------ AGENT ------------------
def maintenance_agent(health_score, failure_probability):
    if failure_probability > 0.8:
        return "IMMEDIATE_MAINTENANCE"
    elif health_score < 50:
        return "SCHEDULE_MAINTENANCE"
    else:
        return "CONTINUE"

# ------------------ LOAD & TRAIN MODEL ------------------
data = pd.read_csv("data/vehicle_data.csv")

X = data[["engine_temp", "rpm", "vibration"]]
y = data["failed"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# ------------------ MEMORY ------------------
memory_file = "data/agent_memory.csv"

# ------------------ SIMULATION LOOP ------------------
print("🚗 Vehicle Health Agent Started...\n")

for step in range(10):  # simulate 10 time steps
    # Simulated sensor data
    temp = random.randint(85, 115)
    rpm = random.randint(2500, 4500)
    vibration = round(random.uniform(0.02, 0.10), 2)

    # Observe
    health = calculate_health(temp, rpm, vibration)
    failure_prob = model.predict_proba([[temp, rpm, vibration]])[0][1]

    # Decide
    decision = maintenance_agent(health, failure_prob)

    # Simulate outcome
    outcome = "FAILURE" if failure_prob > 0.7 else "OK"

    # Save to memory
    record = pd.DataFrame([{
        "engine_temp": temp,
        "rpm": rpm,
        "vibration": vibration,
        "health": health,
        "failure_prob": round(failure_prob, 2),
        "decision": decision,
        "outcome": outcome
    }])

    record.to_csv(memory_file, mode="a", header=False, index=False)

    # Act (print)
    print(f"Step {step+1}")
    print(f"Temp={temp}, RPM={rpm}, Vib={vibration}")
    print(f"Health={health}, Failure Risk={round(failure_prob,2)}")
    print(f"Agent Decision → {decision}")
    print(f"Outcome → {outcome}")
    print("-" * 40)

    time.sleep(1)
