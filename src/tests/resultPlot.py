import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Step 1: Prepare the data from the CSV and other trackers
data = {
    'Tracker': [
        'SORT-3D Eucl.', 'SORT-3D IoU', 'SORT', 'DeepSORT', 'OC-SORT', 'FairMOT'
    ],
    'HOTA (%)': [28.35, 7.92, 43.1, 55.0, 54.2, 59.3],
    'MOTA (%)': [21.18, -26.27, 43.1, 61.4, 63.2, 73.7],
    'IDF1 (%)': [35.96, 3.35, 39.8, 62.2, 62.1, 72.3],
    'IDs': [252, 1581, 4852, 781, 522, 330]
}


df = pd.DataFrame(data)

# Step 2: Create separate bar charts for each metric
metrics = ['HOTA (%)', 'MOTA (%)', 'IDF1 (%)', 'IDs']
for metric in metrics:
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Tracker', y=metric, data=df)
    plt.title(f'Comparison of {metric} Across Trackers')
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{metric.lower()}_comparison.png')
    plt.close()

# Step 3: Normalize data for radar chart (except FPS, which is excluded)
df_normalized = df.copy()

# Normalize each column to a 0-1 range
for metric in metrics:
    min_val = df[metric].min()
    max_val = df[metric].max()
    df_normalized[metric] = (df[metric] - min_val) / (max_val - min_val)

# Prepare data for radar chart
labels = metrics
num_vars = len(labels)

# Compute angles for each metric
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

# Create radar chart
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# Plot each tracker
for i, tracker in enumerate(df['Tracker']):
    values = df_normalized.loc[i, metrics].tolist()
    values += values[:1]  # Repeat first value to close the circular graph
    ax.plot(angles + [angles[0]], values, label=tracker, linewidth=2)
    ax.fill(angles + [angles[0]], values, alpha=0.2)

# Format radar chart
ax.set_xticks(angles)
ax.set_xticklabels(labels)
ax.set_yticklabels([])  # Hide radial labels for clarity
plt.title("Radar Chart of Tracking Metrics")
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
plt.tight_layout()
plt.savefig('tracking_metrics_radar.png')
plt.close()

print("Plots have been saved for each metric separately and the radar chart is saved as 'tracking_metrics_radar.png'.")