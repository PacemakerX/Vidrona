import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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

# Time-Based Analysis: Extract useful time features
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

# Visualize Tower-wise Average Sensor Values
plt.figure(figsize=(12, 8))
sns.barplot(x='sensor_id', y='avg_current', data=tower_summary, palette='viridis')
plt.title('Average Sensor Current by Tower', fontsize=16)
plt.xlabel('Tower', fontsize=12)
plt.ylabel('Average Sensor Current', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('../store/avg_current_by_tower.png', dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(12, 8))
sns.barplot(x='sensor_id', y='avg_humidity', data=tower_summary, palette='coolwarm')
plt.title('Average Humidity by Tower', fontsize=16)
plt.xlabel('Tower', fontsize=12)
plt.ylabel('Average Humidity (%)', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('../store/avg_humidity_by_tower.png', dpi=300, bbox_inches='tight')
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
sns.barplot(x='sensor_id', y='avg_current', hue='risk_flag', data=risk_severity, palette='RdYlGn_r')
plt.title('Sensor Current by Tower and Risk Severity', fontsize=16)
plt.xlabel('Tower', fontsize=12)
plt.ylabel('Average Sensor Current', fontsize=12)
plt.xticks(rotation=45)
plt.legend(title='Risk Flag', labels=['Low Risk (0)', 'High Risk (1)'])
plt.tight_layout()
plt.savefig('../store/risk_severity_by_tower.png', dpi=300, bbox_inches='tight')
plt.show()

# **Correlation Analysis**
print("\nCorrelation between different variables:")

# Select numeric columns for correlation analysis
# Exclude hour, day, month to focus on sensor readings
numeric_df = df[['current', 'risk_flag', 'humidity', 'temperature', 'wind_speed', 'voltage']]

# Calculate the correlation matrix
correlation_matrix = numeric_df.corr()
print(correlation_matrix)

# Improved heatmap visualization of correlations
plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)

sns.heatmap(
    correlation_matrix, 
    annot=True,
    cmap=cmap,
    fmt='.2f', 
    vmin=-1, 
    vmax=1, 
    center=0, 
    linewidths=0.5,
    square=True,
    mask=mask
)
plt.title('Correlation Heatmap', fontsize=16)
plt.tight_layout()
plt.savefig('../store/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot lower triangle heatmap with enhanced styling
plt.figure(figsize=(12, 10))
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='coolwarm',
    fmt='.2f',
    vmin=-1,
    vmax=1,
    center=0,
    linewidths=1,
    square=True,
    cbar_kws={"shrink": 0.8},
    annot_kws={"size": 12}
)
plt.title('Sensor Variables Correlation Heatmap', fontsize=18)
plt.xticks(fontsize=12, rotation=45, ha='right')
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig('../store/enhanced_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# NEW ANALYSIS: Time-based patterns
print("\nTime-based Pattern Analysis:")

# Hourly patterns
hourly_patterns = df.groupby('hour').agg(
    avg_current=('current', 'mean'),
    avg_temperature=('temperature', 'mean'),
    avg_humidity=('humidity', 'mean'),
    risk_frequency=('risk_flag', 'mean')
).reset_index()

# Visualize hourly patterns
plt.figure(figsize=(14, 8))
plt.subplot(2, 2, 1)
sns.lineplot(x='hour', y='avg_current', data=hourly_patterns, marker='o', linewidth=2)
plt.title('Average Current by Hour of Day', fontsize=14)
plt.xlabel('Hour', fontsize=12)
plt.ylabel('Current', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(range(0, 24, 2))

plt.subplot(2, 2, 2)
sns.lineplot(x='hour', y='avg_temperature', data=hourly_patterns, marker='o', linewidth=2, color='red')
plt.title('Average Temperature by Hour of Day', fontsize=14)
plt.xlabel('Hour', fontsize=12)
plt.ylabel('Temperature', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(range(0, 24, 2))

plt.subplot(2, 2, 3)
sns.lineplot(x='hour', y='avg_humidity', data=hourly_patterns, marker='o', linewidth=2, color='green')
plt.title('Average Humidity by Hour of Day', fontsize=14)
plt.xlabel('Hour', fontsize=12)
plt.ylabel('Humidity', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(range(0, 24, 2))

plt.subplot(2, 2, 4)
sns.lineplot(x='hour', y='risk_frequency', data=hourly_patterns, marker='o', linewidth=2, color='purple')
plt.title('Risk Frequency by Hour of Day', fontsize=14)
plt.xlabel('Hour', fontsize=12)
plt.ylabel('Risk Frequency', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(range(0, 24, 2))

plt.tight_layout()
plt.savefig('../store/hourly_patterns.png', dpi=300, bbox_inches='tight')
plt.show()

# NEW ANALYSIS: Clustering Analysis
print("\nClustering Analysis:")

# Select features for clustering
features = df[['current', 'humidity', 'temperature', 'wind_speed', 'voltage']]

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Determine optimal number of clusters using the elbow method
wcss = []
for i in range(1, min(8, len(scaled_features))):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, min(8, len(scaled_features))), wcss, marker='o', linestyle='-')
plt.title('Elbow Method for Optimal k', fontsize=16)
plt.xlabel('Number of Clusters', fontsize=12)
plt.ylabel('WCSS', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../store/elbow_method.png', dpi=300, bbox_inches='tight')
plt.show()

# Apply K-means clustering with the optimal number of clusters (assuming 3 from visual inspection)
optimal_clusters = 3
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=42)
df['cluster'] = kmeans.fit_predict(scaled_features)

# Visualize clusters using PCA for dimensionality reduction
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_features)
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['cluster'] = df['cluster']
pca_df['risk_flag'] = df['risk_flag']

plt.figure(figsize=(12, 8))
sns.scatterplot(x='PC1', y='PC2', hue='cluster', data=pca_df, palette='viridis', s=100, alpha=0.7)
plt.title('Sensor Data Clusters (PCA)', fontsize=16)
plt.xlabel('Principal Component 1', fontsize=12)
plt.ylabel('Principal Component 2', fontsize=12)
plt.legend(title='Cluster')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../store/cluster_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Analyze clusters
cluster_analysis = df.groupby('cluster').agg(
    avg_current=('current', 'mean'),
    avg_humidity=('humidity', 'mean'),
    avg_temperature=('temperature', 'mean'),
    avg_wind_speed=('wind_speed', 'mean'),
    avg_voltage=('voltage', 'mean'),
    risk_ratio=('risk_flag', 'mean'),
    count=('sensor_id', 'count')
).reset_index()

print("\nCluster Characteristics:")
print(cluster_analysis)

# Radar chart for cluster profiles
def radar_chart(df, cat_col):
    # Prepare the data
    categories = list(df.columns)[1:-1]  # Exclude cluster and count columns
    N = len(categories)
    
    # Create angles for each category
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # For each cluster
    for i, cluster in enumerate(df[cat_col].unique()):
        values = df[df[cat_col] == cluster].iloc[0, 1:-1].values.tolist()
        values += values[:1]  # Close the loop
        
        # Plot the cluster
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=f'Cluster {cluster}')
        ax.fill(angles, values, alpha=0.1)
    
    # Set category labels
    plt.xticks(angles[:-1], categories, size=12)
    
    # Add legend and title
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Cluster Profiles', size=20, y=1.1)
    
    return fig, ax

# Normalize cluster_analysis for radar chart
cluster_radar = cluster_analysis.copy()
for col in cluster_radar.columns[1:-1]:  # Exclude cluster and count
    cluster_radar[col] = (cluster_radar[col] - cluster_radar[col].min()) / (cluster_radar[col].max() - cluster_radar[col].min())

# Create radar chart
radar_fig, radar_ax = radar_chart(cluster_radar, 'cluster')
plt.tight_layout()
plt.savefig('../store/cluster_radar.png', dpi=300, bbox_inches='tight')
plt.show()

# NEW ANALYSIS: Risk Prediction Model
print("\nRisk Prediction Model:")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Prepare features and target
X = df[['current', 'humidity', 'temperature', 'wind_speed', 'voltage']]
y = df['risk_flag']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
print("\nRandom Forest Model Performance:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix', fontsize=16)
plt.xlabel('Predicted Labels', fontsize=12)
plt.ylabel('True Labels', fontsize=12)
plt.xticks([0.5, 1.5], ['No Risk', 'Risk'])
plt.yticks([0.5, 1.5], ['No Risk', 'Risk'])
plt.tight_layout()
plt.savefig('../store/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis')
plt.title('Feature Importance for Risk Prediction', fontsize=16)
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../store/feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# Save cleaned data for further use
df.to_csv('../store/cleaned_sensor_data.csv', index=False)

print("\nAnalysis Completed!")