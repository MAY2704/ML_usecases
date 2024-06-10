# Import libraries
import pandas as pd
from sklearn.ensemble import IsolationForest
from matplotlib import pyplot as plt

# Define CSV file path (replace with your actual path)
csv_file = "your_data.csv"

# Read data from CSV using pandas
data = pd.read_csv(csv_file)

# Extract features (assuming first two columns contain features)
features = data.iloc[:, :2]  # Select first two columns

# Convert pandas DataFrame to NumPy array (IsolationForest prefers NumPy arrays)
data_array = features.to_numpy()

# Define the Isolation Forest model
model = IsolationForest(contamination=0.1)  # Contamination rate (anomaly proportion)

# Train the model
model.fit(data_array)

# Predict scores (higher score indicates higher anomaly probability)
scores = model.decision_function(data_array)

# Threshold for anomaly detection (adjust based on your data)
threshold = -0.5

# Detect anomalies based on scores and threshold
anomalies = data_array[scores < threshold]

# Extract original data points corresponding to anomalies
anomalous_data = data.iloc[np.where(scores < threshold)[0]]

# Plot the data with anomalies highlighted
plt.scatter(data_array[:, 0], data_array[:, 1], color='blue')
plt.scatter(anomalies[:, 0], anomalies[:, 1], color='red')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Anomaly Detection with Isolation Forest")
plt.show()

# Print the detected anomalies (including their corresponding data from the CSV)
print("Anomalies:")
for index, row in anomalous_data.iterrows():
  print(f"Data Point: {row.to_numpy()}")

