import boto3
import json
from decimal import Decimal
from datetime import datetime
import pytz

# Initialize DynamoDB client
dynamodb = boto3.resource('dynamodb', region_name='ap-south-1')  # Update region if needed
table = dynamodb.Table('ElectricLines')  # Table name

# Function to convert Decimal to float for JSON serialization
def decimal_to_float(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError

# Function to convert UTC timestamp to IST (without milliseconds)
def convert_to_ist(utc_timestamp):
    # Convert UTC timestamp to datetime object (handle milliseconds with .%f)
    utc_time = datetime.strptime(utc_timestamp, '%Y-%m-%dT%H:%M:%S.%fZ')  # Handle milliseconds using %f
    # Set timezone to UTC
    utc_time = utc_time.replace(tzinfo=pytz.UTC)
    # Convert to IST (Indian Standard Time)
    ist_time = utc_time.astimezone(pytz.timezone('Asia/Kolkata'))
    # Format it as string again in the desired format (no milliseconds)
    return ist_time.strftime('%Y-%m-%dT%H:%M:%SZ')

def fetch_data(sensor_id=None, start_timestamp=None, end_timestamp=None):
    try:
        # Building filter expressions
        expression = 'sensor_id = :sensor_id' if sensor_id else 'timestamp BETWEEN :start_timestamp AND :end_timestamp'
        expression_values = {}

        if sensor_id:
            expression_values = {':sensor_id': sensor_id}
        else:
            expression_values = {':start_timestamp': start_timestamp, ':end_timestamp': end_timestamp}

        # Fetch data from DynamoDB based on provided filters
        response = table.scan(
            FilterExpression=expression,
            ExpressionAttributeValues=expression_values
        )

        return response['Items']
    
    except Exception as e:
        print(f"Error fetching data from DynamoDB: {e}")
        return []

# Test with specific sensor ID
data = fetch_data(sensor_id='tower_9')

# Convert Decimal values to float and change timestamps to IST for JSON serialization
for item in data:
    item['timestamp'] = convert_to_ist(item['timestamp'])

# Print the fetched data with IST timestamp
print(f"Fetched Data: {json.dumps(data, default=decimal_to_float, indent=2)}")
