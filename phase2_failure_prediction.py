import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
data = pd.read_csv("data/vehicle_data.csv")

X = data[["engine_temp", "rpm", "vibration"]]
y = data["failed"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = RandomForestClassifier(random_state=42)

# Train model
model.fit(X_train, y_train)

# Test model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)

# =========================
# 🧱 STEP 6: Predict Failure for New Vehicle
# =========================

# Predict for a new vehicle
new_vehicle = [[107, 3900, 0.08]]

failure_prediction = model.predict(new_vehicle)
failure_probability = model.predict_proba(new_vehicle)

print("\nNew Vehicle Prediction:")
print("Will Fail (0=No, 1=Yes):", failure_prediction[0])
print("Failure Probability:", failure_probability[0][1])
