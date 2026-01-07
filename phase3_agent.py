import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ---------- Health Score Logic (Phase 1) ----------
def calculate_health(temp, rpm, vibration):
    score = 100

    if temp > 95:
        score -= 20
    if rpm > 3500:
        score -= 20
    if vibration > 0.05:
        score -= 40

    return max(score, 0)

# ---------- Agent Decision Logic (Phase 3) ----------
def maintenance_agent(health_score, failure_probability):
    if failure_probability > 0.8:
        return "IMMEDIATE MAINTENANCE REQUIRED"
    elif health_score < 50:
        return "SCHEDULE MAINTENANCE SOON"
    else:
        return "VEHICLE OK - CONTINUE OPERATION"

# ---------- Load & Train ML Model (Phase 2) ----------
data = pd.read_csv("data/vehicle_data.csv")

X = data[["engine_temp", "rpm", "vibration"]]
y = data["failed"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# ---------- NEW VEHICLE OBSERVATION ----------
new_vehicle = {
    "engine_temp": 108,
    "rpm": 3900,
    "vibration": 0.08
}

# Observe
health = calculate_health(
    new_vehicle["engine_temp"],
    new_vehicle["rpm"],
    new_vehicle["vibration"]
)

failure_prob = model.predict_proba([[
    new_vehicle["engine_temp"],
    new_vehicle["rpm"],
    new_vehicle["vibration"]
]])[0][1]

# Decide
decision = maintenance_agent(health, failure_prob)

# Act
print("🔍 VEHICLE OBSERVATION")
print(new_vehicle)

print("\n📊 HEALTH & RISK")
print("Health Score:", health)
print("Failure Probability:", round(failure_prob, 2))

print("\n🤖 AGENT DECISION")
print(decision)
