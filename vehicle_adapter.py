def get_vehicle_config(vehicle_type):
    if vehicle_type == "car":
        return {
            "temp_limit": 95,
            "rpm_limit": 3500,
            "failure_cost": 1000
        }

    elif vehicle_type == "truck":
        return {
            "temp_limit": 110,
            "rpm_limit": 3000,
            "failure_cost": 5000
        }

    elif vehicle_type == "aircraft":
        return {
            "temp_limit": 900,  # EGT
            "rpm_limit": None,
            "failure_cost": 100000
        }

    elif vehicle_type == "bike":
        return {
            "temp_limit": 105,
            "rpm_limit": 8000,
            "failure_cost": 3000
        }

    elif vehicle_type == "scooter":
        return {
            "temp_limit": 95,
            "rpm_limit": 7000,
            "failure_cost": 1500
        }

    else:
        raise ValueError("Unknown vehicle type")
