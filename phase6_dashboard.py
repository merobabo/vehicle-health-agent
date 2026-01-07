import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from stable_baselines3 import PPO

from logger import log_event, log_warning
from vehicle_adapter import get_vehicle_config
from fleet_rl_helper import fleet_rl_decision

# ---------------- PAGE SETUP ----------------
st.set_page_config(page_title="Fleet RL Vehicle Agent", layout="centered")
st.title("🚘 Fleet-Level Reinforcement Learning Maintenance Agent")

st.write("""
Compare **Rule-Based**, **Single-Vehicle RL**, and **Fleet-Level RL** decisions.
""")

# ---------------- FAILURE MODEL ----------------
@st.cache_data
def train_failure_model():
    data = pd.read_csv("data/vehicle_data.csv")
    X = data[["engine_temp", "rpm", "vibration"]]
    y = data["failed"]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

failure_model = train_failure_model()

# ---------------- SINGLE RL AGENT ----------------
@st.cache_resource
def load_single_rl():
    return PPO.load("rl_maintenance_agent")

single_rl = load_single_rl()

# ---------------- LOGIC ----------------
def calculate_health(temp, rpm, vibration, config):
    score = 100
    if temp > config["temp_limit"]:
        score -= 30
    if config["rpm_limit"] and rpm > config["rpm_limit"]:
        score -= 20
    if vibration > 0.05:
        score -= 40
    return max(score, 0)

def rule_agent(health, failure_prob, cost):
    expected_loss = failure_prob * cost
    if expected_loss > 20000:
        return "🚨 IMMEDIATE MAINTENANCE"
    elif expected_loss > 5000 or health < 50:
        return "⚠️ SCHEDULE MAINTENANCE"
    else:
        return "✅ CONTINUE"

def single_rl_decision(health, failure_prob):
    state = np.array([[health, failure_prob]], dtype=np.float32)
    action, _ = single_rl.predict(state)
    action = int(action)
    return ["✅ CONTINUE", "⚠️ SCHEDULE MAINTENANCE", "🚨 IMMEDIATE MAINTENANCE"][action]

# ---------------- VEHICLE INPUT ----------------
vehicle_type = st.selectbox(
    "Select Vehicle Type",
    ["car", "truck", "aircraft", "bike", "scooter"]
)

config = get_vehicle_config(vehicle_type)

st.subheader("🔧 Sensor Inputs")

if vehicle_type == "aircraft":
    temp = st.slider("Engine Temperature (EGT)", 600, 950, 750)
    rpm = 0
else:
    temp = st.slider("Engine Temperature (°C)", 70, 130, 90)
    rpm = st.slider("RPM", 1000, 9000, 3000)

vibration = st.slider("Vibration", 0.01, 0.12, 0.03)

# ---------------- PIPELINE ----------------
health = calculate_health(temp, rpm, vibration, config)

failure_prob = failure_model.predict_proba(
    [[temp, rpm, vibration]]
)[0][1]

rule_action = rule_agent(health, failure_prob, config["failure_cost"])
single_rl_action = single_rl_decision(health, failure_prob)

# --- Fleet state (simulate 3 vehicles)
fleet_state = [
    (health, failure_prob),
    (np.random.randint(30, 100), round(np.random.random(), 2)),
    (np.random.randint(30, 100), round(np.random.random(), 2)),
]

fleet_actions = fleet_rl_decision(fleet_state)

# ---------------- LOGGING (CORRECT PLACE) ----------------
log_event(
    f"Vehicle={vehicle_type}, "
    f"Health={health}, "
    f"FailureProb={round(failure_prob,2)}, "
    f"RuleAction={rule_action}, "
    f"SingleRL={single_rl_action}, "
    f"FleetActions={fleet_actions}"
)

if failure_prob > 0.85:
    log_warning(f"HIGH RISK detected for {vehicle_type}")

# ---------------- DISPLAY ----------------
st.subheader("📊 Vehicle Status")

st.metric("Health Score", health)
st.metric("Failure Risk", f"{round(failure_prob * 100, 2)} %")

st.subheader("🧠 Decision Comparison")

st.write("### Rule-Based Agent")
st.info(rule_action)

st.write("### Single-Vehicle RL Agent")
st.success(single_rl_action)

st.write("### Fleet-Level RL Agent (3 Vehicles)")
for i, act in enumerate(fleet_actions, 1):
    st.write(f"Vehicle {i}: {act}")

st.subheader("📘 Why Fleet RL Is Better")
st.write("""
Fleet RL considers **system-wide effects**:
- Prevents too many vehicles going into maintenance at once
- Prioritizes high-risk vehicles
- Optimizes long-term fleet availability
""")
