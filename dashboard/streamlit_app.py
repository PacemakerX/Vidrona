import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import boto3
import os   

# Set Streamlit layout
st.set_page_config(layout="wide")
st.title("🔧 Real-Time IoT Monitoring Dashboard")
st.markdown("---")

# Load AWS config
import json
with open("../config/aws_config.json") as f:
    config = json.load(f)

# Connect to DynamoDB
dynamodb = boto3.resource(
    'dynamodb',
    region_name=config["region"],
    aws_access_key_id=config["aws_access_key_id"],
    aws_secret_access_key=config["aws_secret_access_key"]
)

table = dynamodb.Table(config["table_name"])

# Fetch data from DynamoDB
def fetch_data():
    response = table.scan()
    data = response['Items']
    while 'LastEvaluatedKey' in response:
        response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
        data.extend(response['Items'])
    return data

# Fetch and prepare dataframe
raw_data = fetch_data()
if not raw_data:
    st.warning("🚨 No data found in DynamoDB!")
    st.stop()

df = pd.DataFrame(raw_data)

# Convert and sort timestamp
df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
df = df.sort_values(by='timestamp')
df['readable_time'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

# Convert sensor fields to float if needed
for col in ['current', 'voltage', 'temperature', 'humidity', 'wind_speed', 'risk_flag']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Show raw data toggle
with st.expander("📊 Show Raw Data"):
    st.dataframe(df.tail(20)[['readable_time', 'current', 'voltage', 'temperature', 'humidity']], use_container_width=True)

# KPIs
st.subheader("📈 Key Sensor Stats (Last 10 Rows)")
latest = df.tail(10)
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("⚡ Avg Current (A)", f"{latest['current'].mean():.2f}")
with col2:
    st.metric("🔋 Avg Voltage (V)", f"{latest['voltage'].mean():.2f}")
with col3:
    st.metric("🌡️ Avg Temp (°C)", f"{latest['temperature'].mean():.2f}")
with col4:
    st.metric("💧 Avg Humidity (%)", f"{latest['humidity'].mean():.2f}")

# Line plots
st.markdown("### 📉 Sensor Trends Over Time")
fig, ax = plt.subplots(figsize=(14, 4))
for column in ['current', 'voltage', 'temperature', 'humidity']:
    ax.plot(df['timestamp'], df[column], label=column)
ax.set_xlabel("Timestamp")
ax.set_ylabel("Sensor Values")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Correlation heatmap
st.markdown("### 🔍 Correlation Heatmap Between Variables")
corr = df[['current', 'voltage', 'humidity', 'temperature', 'wind_speed', 'risk_flag']].corr()
fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax2)
st.pyplot(fig2)

# Footer
st.markdown("---")
st.markdown("✅ Dashboard now fetching live data from DynamoDB.")
