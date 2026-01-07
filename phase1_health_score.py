import pandas as pd

def calculate_health(temp, rpm, vibration):
    score = 100

    if temp > 95:
        score -= 20
    if rpm > 3500:
        score -= 20
    if vibration > 0.05:
        score -= 40

    return max(score, 0)

# Load data
data = pd.read_csv("data/vehicle_data.csv")

# Calculate health for each row
print("\nVehicle Health Report:")
for index, row in data.iterrows():
    health = calculate_health(
        row["engine_temp"],
        row["rpm"],
        row["vibration"]
    )
    print(f"Row {index+1} → Health Score: {health}")
