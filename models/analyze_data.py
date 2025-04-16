import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data from the CSV
df = pd.read_csv('../store/sensor_data_for_analysis.csv')

# Display the first few rows of the dataset
print("Data Sample:")
print(df.head())

# Data Cleaning
print("\nCleaning Data...")
# Convert timestamp to datetime format
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%dT%H:%M:%SZ')

# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing Values in each column:")
print(missing_values)

# Separate numeric and non-numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns

# Fill missing values in numeric columns with the mean of each column
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# For non-numeric columns, fill missing values with the mode (most frequent value)
for col in non_numeric_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Check if all missing values are handled
print("\nData after cleaning missing values:")
print(df.isnull().sum())

# Descriptive Statistics
print("\nDescriptive Statistics:")
desc_stats = df.describe()
print(desc_stats)

# Time-Based Analysis: Extract useful time features (optional)
df['hour'] = df['timestamp'].dt.hour
df['day'] = df['timestamp'].dt.dayofweek
df['month'] = df['timestamp'].dt.month

# **Tower-by-Tower Analysis**
print("\nTower-by-Tower Analysis:")

# Group data by 'sensor_id' (representing towers) and calculate summary statistics
tower_summary = df.groupby('sensor_id').agg(
    avg_current=('current', 'mean'),
    avg_humidity=('humidity', 'mean'),
    avg_temperature=('temperature', 'mean'),
    avg_wind_speed=('wind_speed', 'mean'),
    avg_voltage=('voltage', 'mean'),
    total_risk_flags=('risk_flag', 'sum')
).reset_index()

print(tower_summary)

# Visualize Tower-wise Average Sensor Values (e.g., 'current' and 'humidity')
plt.figure(figsize=(12, 8))
sns.barplot(x='sensor_id', y='avg_current', data=tower_summary, palette='viridis')
plt.title('Average Sensor Current by Tower')
plt.xlabel('Tower')
plt.ylabel('Average Sensor Current')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
sns.barplot(x='sensor_id', y='avg_humidity', data=tower_summary, palette='coolwarm')
plt.title('Average Humidity by Tower')
plt.xlabel('Tower')
plt.ylabel('Average Humidity (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# **Risk Severity Analysis**
print("\nRisk Severity Analysis:")

# Define a risk severity factor based on the 'risk_flag' (1 = High risk, 0 = Low risk)
risk_severity = df.groupby(['sensor_id', 'risk_flag']).agg(
    avg_current=('current', 'mean'),
    avg_temperature=('temperature', 'mean'),
    avg_wind_speed=('wind_speed', 'mean')
).reset_index()

# Visualizing Risk Severity by Tower
plt.figure(figsize=(12, 8))
sns.barplot(x='sensor_id', y='avg_current', hue='risk_flag', data=risk_severity, palette='RdYlGn')
plt.title('Sensor Current by Tower and Risk Severity')
plt.xlabel('Tower')
plt.ylabel('Average Sensor Current')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# **Correlation Analysis**
print("\nCorrelation between different variables:")

# Select only numeric columns
numeric_df = df.select_dtypes(include=[np.number])

# Calculate the correlation matrix
correlation_matrix = numeric_df.corr()
print(correlation_matrix)

# Heatmap visualization of correlations with improved scaling
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1, center=0, linewidths=0.5)
plt.title('Correlation Heasatmap (Properly Scaled)')
plt.tight_layout()
plt.show()

# Save cleaned data for further use
df.to_csv('../store/cleaned_sensor_data.csv', index=False)

print("\nAnalysis Completed!")
