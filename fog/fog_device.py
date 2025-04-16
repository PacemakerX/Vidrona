import paho.mqtt.client as mqtt
import json
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
    try:
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
            
            # Publish the filtered (processed) data to the output topic with message retention
            client.publish(output_topic, json.dumps(processed_data), retain=True)
            print(f"Processed Data Sent: {processed_data}")
        else:
            print("Data filtered out due to threshold mismatch.")
    except json.JSONDecodeError:
        print("Failed to decode message as JSON.")
    except Exception as e:
        print(f"Error processing message: {e}")

# Callback for handling connection success
def on_connect(client, userdata, flags, rc):
    print(f"Connected to broker with result code {rc}")
    # Subscribe to the input topic after connection is successful
    client.subscribe(input_topic)

# Callback for handling connection loss (reconnect if necessary)
def on_disconnect(client, userdata, rc):
    print(f"Disconnected from broker with result code {rc}")
    if rc != 0:
        print("Unexpected disconnection. Attempting to reconnect...")
        client.reconnect()

# Setup MQTT client and connect to the broker
client = mqtt.Client()

# Set the callback functions
client.on_message = on_message
client.on_connect = on_connect
client.on_disconnect = on_disconnect

# Connect to the broker
try:
    client.connect(broker, port)
except Exception as e:
    print(f"Failed to connect to broker: {e}")
    exit(1)

# Start the MQTT loop to handle incoming messages
client.loop_start()

# Keep the fog device running to listen for incoming messages
try:
    while True:
        time.sleep(1)  # Wait to keep the program running
except KeyboardInterrupt:
    print("Fog device stopped.")
    client.loop_stop()
