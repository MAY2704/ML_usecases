# Import libraries
import numpy as np
from sklearn.ensemble import IsolationForest
from matplotlib import pyplot as plt

# Sample data (replace with your actual data)
data = np.array([
    [1, 2],
    [3, 4],
    [1.5, 1.8],
    [5, 8],
    [6, 999],  # Outlier
    [4.5, 5],
    [7, 8.5],
    [8, 9],
])

# Define the Isolation Forest model
model = IsolationForest(contamination=0.1)  # Contamination rate (anomaly proportion)

# Train the model
model.fit(data)

# Predict scores (higher score indicates higher anomaly probability)
scores = model.decision_function(data)

# Threshold for anomaly detection (adjust based on your data)
threshold = -0.5

# Detect anomalies based on scores and threshold
anomalies = data[scores < threshold]

# Plot the data with anomalies highlighted
plt.scatter(data[:, 0], data[:, 1], color='blue')
plt.scatter(anomalies[:, 0], anomalies[:, 1], color='red')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Anomaly Detection with Isolation Forest")
plt.show()

# Print the detected anomalies
print("Anomalies:")
for anomaly in anomalies:
  print(anomaly)

