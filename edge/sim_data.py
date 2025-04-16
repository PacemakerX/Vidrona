# edge/sim_data.py

import paho.mqtt.client as mqtt
import json
import random
import time
from datetime import datetime

broker = "localhost"  # Use your EC2 public IP or Mosquitto IP if on AWS
port = 1883
topic = "iot/electricity"

client = mqtt.Client()
client.connect(broker, port)

LOCATIONS = ["Sector-21", "GreenValley", "Hilltop-Grid", "Downtown", "WestTower"]

def simulate_sensor_data():
    return {
        "sensor_id": f"tower_{random.randint(1, 10)}",
        "location": random.choice(LOCATIONS),
        "voltage": round(random.uniform(210, 250), 2),      # in volts
        "current": round(random.uniform(5, 15), 2),          # in amperes
        "temperature": round(random.uniform(25, 50), 2),     # in °C
        "humidity": round(random.uniform(30, 90), 2),        # in %
        "wind_speed": round(random.uniform(0, 40), 2),       # in km/h
        "risk_flag": random.choice([0, 1]),
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

while True:
    data = simulate_sensor_data()
    client.publish(topic, json.dumps(data))
    print("Sent:", data)
    time.sleep(3)  # Send every 3 seconds
