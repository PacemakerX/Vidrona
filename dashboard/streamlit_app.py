import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set Streamlit layout
st.set_page_config(layout="wide")
st.title("🔧 Real-Time IoT Monitoring Dashboard")
st.markdown("---")

# Set CSV path (make sure your analyze_data.py stores here)
DATA_DIR = "../store/"
CSV_FILE = os.path.join(DATA_DIR, "sensor_data_for_analysis.csv")
CORR_IMG = os.path.join(DATA_DIR, "correlation_heatmap.png")

# Check if data is available
if not os.path.exists(CSV_FILE):
    st.warning("🚨 No data found! Please run `analyze_data.py` to generate data.")
    st.stop()

df = pd.read_csv(CSV_FILE)
df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)  # retain UTC for real-time logic
df = df.sort_values(by='timestamp')  # ensure sorted


# Show raw data toggle
with st.expander("📊 Show Raw Data"):
    st.dataframe(df.tail(20), use_container_width=True)

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

if os.path.exists(CORR_IMG):
    st.image(CORR_IMG, caption="Correlation Heatmap", use_container_width=True)
else:
    # Fallback live heatmap if no PNG exists
    corr = df[['current', 'voltage', 'humidity', 'temperature', 'wind_speed', 'risk_flag']].corr()
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)

# Footer
st.markdown("---")
st.markdown("✅ Dashboard running locally. Connect AWS DynamoDB later for live data.")