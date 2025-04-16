import paho.mqtt.client as mqtt
import json
import boto3
from decimal import Decimal

# Initialize AWS DynamoDB resource
dynamodb = boto3.resource('dynamodb', region_name='ap-south-1')  # Use your region
table = dynamodb.Table('ElectricLines')  # Use your table name

# MQTT Broker settings
broker = "localhost"  # Use your EC2 public IP or Mosquitto IP if on AWS
port = 1883
topic = "processed/electricity"  # Processed data topic

# MQTT client initialization
client = mqtt.Client(client_id="dynamodb_writer", protocol=mqtt.MQTTv5)  # Update to MQTTv5 to avoid deprecation warning

# Callback to handle received data
def on_message(client, userdata, message):
    # Add this code temporarily to your script to print table details
    table_description = dynamodb.meta.client.describe_table(TableName='ElectricLines')
    print("Table structure:", json.dumps(table_description, indent=2, default=str))
    try:
        sensor_data = json.loads(message.payload)
        
        # Sanitize & validate sensor_id
        raw_id = sensor_data.get("sensor_id", "")
        sensor_id = str(raw_id).strip()  # remove whitespace + ensure string
        
        if not sensor_id or 'timestamp' not in sensor_data:
            print(f"❌ Invalid data | sensor_id: {repr(sensor_id)} | Full payload: {sensor_data}")
            return
        
        item = {
            'sensor_id': sensor_id,
            'timestamp': str(sensor_data['timestamp']),
            'voltage': Decimal(str(sensor_data['voltage'])),
            'current': Decimal(str(sensor_data['current'])),
            'temperature': Decimal(str(sensor_data['temperature'])),
            'humidity': Decimal(str(sensor_data['humidity'])),
            'wind_speed': Decimal(str(sensor_data['wind_speed'])),
            'risk_flag': int(sensor_data['risk_flag']),
            'location': str(sensor_data['location']).strip()
        }
        
        print("✅ DEBUG | Prepared item:", item)
        
        # Insert directly - boto3 handles type conversion
        response = table.put_item(Item=item)
        print("✅ Data inserted successfully.")
        
    except Exception as e:
        print(f"❌ Error inserting data into DynamoDB: {e}")

# Setup MQTT client
client.on_message = on_message
client.connect(broker, port)

# Subscribe to the topic where processed data is published
client.subscribe(topic)

# Start MQTT loop
client.loop_start()

try:
    while True:
        pass  # Keep the script running
except KeyboardInterrupt:
    print("Writer stopped.")
    client.loop_stop()