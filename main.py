import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv('chemical_compounds.csv')

# Find non-numeric columns
non_numeric_columns = data.columns[(data.dtypes == 'object').values].tolist()

# Print non-numeric columns
#print(f"Non-numeric columns: {non_numeric_columns}")

# Drop the non-numeric 'PUBCHEM_COORDINATE_TYPE' column
data = data.drop('PUBCHEM_COORDINATE_TYPE', axis=1)

# Check for NaN values in the dataset
nan_values = data.isna()

# Print rows with NaN values
#print("Rows with NaN values:")
#print(data[nan_values.any(axis=1)])

# Drop any NaN values
data = data.dropna()

# Find non-numeric columns
non_numeric_columns = data.columns[(data.dtypes == 'object').values].tolist()

# Print non-numeric columns
#print(f"Non-numeric columns: {non_numeric_columns}")

#print("\n \n \n \n \n ")

X = data.drop('Class', axis=1)  # Features of target variable
y = data['Class']  # Target variable

# Split the dataset into training (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initalize SVM classifer with a linear kernal
svm_classifier = SVC(kernel='linear')

# Train the classifier
svm_classifier.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = svm_classifier.predict(X_test_scaled)

# Print out results of predictions agaisnt the test set
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)


print("Accuracy: ", accuracy ,"\n")
print("Class report: ")
print("--------------------------------------------------------")
print(class_report)


# Load the new dataset
new_data = pd.read_csv('new_chemical_compounds.csv')

# Drop non-numeric columns and handle missing values 
new_data = new_data.drop('PUBCHEM_COORDINATE_TYPE', axis=1)
new_data = new_data.dropna()

# Feature Set
X_new = new_data
X_new_scaled = scaler.transform(X_new)  # set scalar

# Set target variable to predict
y_new_pred = svm_classifier.predict(X_new_scaled)

# Add predictions to the new dataset
new_data['Predicted_Class'] = y_new_pred

# Save the results as a csv file 
new_data.to_csv('predicted_braf_inhibitors.csv', index=False)


