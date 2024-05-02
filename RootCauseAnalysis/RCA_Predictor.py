import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import export_graphviz
import pydot

# Sample data defined for feature columnms like environment, poc etc
data = {
       "ENV": [],
       "POC": [],
       "Project": [],
       "Category": [],
       "Assigned_team": [],
       "Root_cause": []
}

# Define expected column order (include features and 
expected_columns = ["ENV", "POC", "Project", "Category", "Assigned_team", "Root_cause"]

# Create DataFrame and handle potential non-string values
try:
  data = pd.DataFrame({col: [str(x) for x in data[col]] for col in expected_columns})
except (ValueError, TypeError) as e:
  print(f"Error creating DataFrame: {e}")
  exit(1)  # Exit with error code

# Encode categorical features
encoder = LabelEncoder()
data["ENV_encoded"] = encoder.fit_transform(data["ENV"])
data["POC_encoded"] = encoder.fit_transform(data["POC"])
data["Project_encoded"] = encoder.fit_transform(data["Project"])
data["CATEGORY_encoded"] = encoder.fit_transform(data["Category"])
data["Assigned_team_encoded"] = encoder.fit_transform(data["Assigned_team"])
data["Root_cause_encoded"] = encoder.fit_transform(data["Root_cause"])

# Separate features and target variable
X = data[["ENV_encoded", "POC_encoded", "Project_encoded", "CATEGORY_encoded","Assigned_team_encoded"]]
y = data["Root_cause_encoded"]
feature_list = list(X.columns)
X=X.astype(int)
y=y.astype(int)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define and train the model
model = RandomForestClassifier(n_estimators=400, random_state=21)
model.fit(X_train, y_train)

# Calculate model accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Ensure y_test and predicted labels (y_pred) are available
y_pred = model.predict(X_test)  # Assuming prediction is done

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Create informative visualization
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)  # Color scheme for clarity
plt.colorbar(shrink=0.5)  # Colorbar for interpreting intensities

# Customize labels and ticks for better readability
class_names = np.unique(y_test)  # Assuming class labels are available
plt.xticks(np.arange(len(class_names)), class_names, rotation=45, ha='right')
plt.yticks(np.arange(len(class_names)), class_names)

# Add informative labels and title
plt.xlabel('Predicted root cause')
plt.ylabel('True root cause')
plt.title('Confusion Matrix')

# Annotate entries with counts for easier visual interpretation
for i in range(len(cm)):
    for j in range(len(cm[0])):
        plt.text(j, i, cm[i, j], ha='center', va='center', fontsize=8)  # Adjust fontsize as needed

plt.tight_layout()
plt.show()


# Pull out one tree from the forest
tree = model.estimators_[5]
# Export the image to a dot file
export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')
# Write grap to a png file
graph.write_png('tree.png')

new_data = {
  "ENV_encoded": 2,  # Replace with the encoded value for the new data's environment
  "POC_encoded": 1,   # Replace with the encoded value for the new data's POC
  "Project_encoded" : 2,
  "CATEGORY_encoded" : 1,
  "Assigned_team_encoded" : 2
  # ... (other encoded features)
}

# Function to make predictions for a new data point
def predict_new_data(data):
  """
  This function takes a dictionary containing encoded categorical features
  and predicts the root cause using the trained Random Forest model.

  Args:
      data (dict): Dictionary containing encoded features (e.g., ENV_encoded, POC_encoded, etc.)
                  for the new data point.

  Returns:
      int: Predicted root cause class label.
  """

  # Ensure the new data has the same features and order as the training data
  new_data_df = pd.DataFrame(columns=feature_list)  # Replace feature_list with your actual feature names
  new_data_df.loc[0] = data  # Add the new data as a single row in the DataFrame

  # Transform the new data (if necessary)
  # ... (apply any necessary transformations like scaling)

  # Predict the root cause for the new data
  new_prediction = model.predict(new_data_df)[0]

  # Decode the predicted label back to the original class name (optional)
  # decoded_prediction = class_names[new_prediction]  # Assuming class labels are available in class_names

  return new_prediction 
predicted_root_cause = predict_new_data(new_data)
print("Predicted root cause:", predicted_root_cause)
