# fog_device.py

import paho.mqtt.client as mqtt
import json
import random
import time

broker = "localhost"  # Use your EC2 public IP or Mosquitto IP if on AWS
port = 1883
input_topic = "iot/electricity"  # Topic for raw sensor data
output_topic = "processed/electricity"  # Topic for processed data

# Define threshold values for data filtering
VOLTAGE_THRESHOLD = 230  # Only process data with voltage > 230V
TEMPERATURE_THRESHOLD = 30  # Only process data with temperature > 30°C

# Callback function for handling received messages
def on_message(client, userdata, message):
    # Parse the message payload (sensor data) from JSON
    sensor_data = json.loads(message.payload)
    
    # Data Filtering Logic: Process only data that meets certain criteria
    if sensor_data["voltage"] > VOLTAGE_THRESHOLD and sensor_data["temperature"] > TEMPERATURE_THRESHOLD:
        # Apply any additional filtering or anomaly detection here if needed
        processed_data = {
            "sensor_id": sensor_data["sensor_id"],
            "location": sensor_data["location"],
            "voltage": sensor_data["voltage"],
            "current": sensor_data["current"],
            "temperature": sensor_data["temperature"],
            "humidity": sensor_data["humidity"],
            "wind_speed": sensor_data["wind_speed"],
            "risk_flag": sensor_data["risk_flag"],
            "timestamp": sensor_data["timestamp"]
        }
        
        # Publish the filtered (processed) data to the output topic
        client.publish(output_topic, json.dumps(processed_data))
        print(f"Processed Data Sent: {processed_data}")
    else:
        print("Data filtered out due to threshold mismatch.")

# Setup MQTT client and connect to the broker
client = mqtt.Client()
client.connect(broker, port)

# Set the callback function for when a message is received
client.on_message = on_message

# Subscribe to the topic where raw sensor data is published
client.subscribe(input_topic)

# Start the MQTT loop to handle incoming messages
client.loop_start()

# Keep the fog device running to listen for incoming messages
try:
    while True:
        time.sleep(1)  # Wait to keep the program running
except KeyboardInterrupt:
    print("Fog device stopped.")
    client.loop_stop()
